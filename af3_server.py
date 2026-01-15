"""AlphaFold 3 inference server (fixed config, queued requests).

This server is intended for RL/reward-loop usage where you want:
  - a persistent process that loads model parameters once
  - a fixed inference config (num_recycles / num_diffusion_samples)
  - server-side request queueing (single GPU worker by default)
  - saving predicted structures (mmCIF) and returning metrics (iptm, pLDDT, ...)

Notes:
  - This server uses empty MSA and no templates (no data pipeline).
  - pLDDT is returned as the mean of `predicted_structure.atom_b_factor`.
    In this codebase, predicted LDDT is computed and multiplied by 100.0 in
    `alphafold3/model/network/confidence_head.py` and then written to B-factors
    in `alphafold3/model/model.py:get_predicted_structure`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import os
import pathlib
import time
import uuid
from typing import Any, Literal, Optional

from absl import app as absl_app
from absl import flags
import jax
import numpy as np
from pydantic import BaseModel, Field

from fastapi import FastAPI

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation

# IMPORTANT:
# `run_alphafold.py` defines a large set of absl flags at import time.
# To keep *all* parameters aligned (and avoid DuplicateFlagError), this server
# reuses those flags instead of redefining them.
import run_alphafold as ra


# Server-only flags.
_HOST = flags.DEFINE_string("host", "0.0.0.0", "Server host to bind to.")
_PORT = flags.DEFINE_integer("port", 8000, "Server port.", lower_bound=1)
_WORKERS = flags.DEFINE_integer(
    "workers",
    1,
    "Number of uvicorn worker processes. Keep 1 to avoid multiple model loads on GPU.",
    lower_bound=1,
)
_SERVER_NUM_GPUS = flags.DEFINE_integer(
    "server_num_gpus",
    None,
    "Number of GPUs to use for multi-GPU serving (single process, shared queue). "
    "If set, the server uses the first N visible GPUs (controlled via CUDA_VISIBLE_DEVICES).",
    lower_bound=1,
)

def _choose_device(gpu_device: int | None) -> jax.Device:
  gpus = jax.local_devices(backend="gpu")
  if gpus:
    idx = 0 if gpu_device is None else int(gpu_device)
    if idx < 0 or idx >= len(gpus):
      raise ValueError(f"gpu_device={idx} out of range, available GPUs: {gpus}")
    return gpus[idx]
  # CPU fallback (not recommended for AF3 inference, but keeps the server usable).
  return jax.local_devices()[0]


def _validate_device_and_attention_impl(device: jax.Device, flash_impl: str) -> None:
  """Mirrors the device/flag compatibility checks in run_alphafold.py."""
  if getattr(device, "platform", None) != "gpu":
    return
  compute_capability = getattr(device, "compute_capability", None)
  if compute_capability is None:
    return
  cc = float(compute_capability)
  if cc < 6.0:
    raise ValueError(
        "AlphaFold 3 requires at least GPU compute capability 6.0 "
        "(see https://developer.nvidia.com/cuda-gpus)."
    )
  elif 7.0 <= cc < 8.0:
    required_flag = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
    xla_flags = os.environ.get("XLA_FLAGS")
    if not xla_flags or required_flag not in xla_flags:
      raise ValueError(
          "For devices with GPU compute capability 7.x the ENV XLA_FLAGS must "
          f'include "{required_flag}".'
      )
    if flash_impl != "xla":
      raise ValueError(
          'For devices with GPU compute capability 7.x the '
          '--flash_attention_implementation must be set to "xla".'
      )

def _mean_plddt_from_structure(pred_structure) -> float:
  # AF3 writes predicted LDDT (0-100) into the structure's atom B-factors.
  b = np.asarray(pred_structure.atom_b_factor, dtype=np.float32)
  return float(b.mean()) if b.size else 0.0


class ChainIn(BaseModel):
  id: str = Field(..., description="Chain ID (single uppercase letter recommended).")
  sequence: str = Field(..., description="Protein sequence (letters).")


RewardName = Literal["iptm", "ranking_score", "ptm_iptm_average", "ptm"]


class PredictRequest(BaseModel):
  name: str = Field("job", description="Job name used for output filenames.")
  seed: int = Field(0, description="RNG seed for inference.")
  chains: list[ChainIn] = Field(..., description="Protein chains for the complex.")
  reward: RewardName = Field("iptm", description="Which scalar to return as reward.")
  return_mmcif_text: bool = Field(
      False, description="If true, also include mmCIF text in the response."
  )


@dataclasses.dataclass(frozen=True, slots=True)
class _ServerConfig:
  model_dir: pathlib.Path
  output_dir: pathlib.Path
  buckets: tuple[int, ...]
  max_template_date: datetime.date
  conformer_max_iterations: int | None
  resolve_msa_overlaps: bool
  flash_attention_implementation: str
  num_diffusion_samples: int
  num_recycles: int
  gpu_device: int
  server_num_gpus: int | None


# Global server state (initialized in FastAPI startup).
app = FastAPI()
_queue: asyncio.Queue[tuple[PredictRequest, asyncio.Future]] = asyncio.Queue()
_model_runners: list[ra.ModelRunner] = []
_server_cfg: Optional[_ServerConfig] = None


async def _worker_loop(worker_idx: int, model_runner: ra.ModelRunner):
  # Worker loop: processes requests sequentially on its bound device.
  while True:
    req, fut = await _queue.get()
    try:
      assert _server_cfg is not None
      out = _predict_one(req, _server_cfg, model_runner)
      fut.set_result(out)
    except Exception as e:  # pylint: disable=broad-exception-caught
      fut.set_exception(e)
    finally:
      _queue.task_done()


def _predict_one(
    req: PredictRequest, cfg: _ServerConfig, model_runner: ra.ModelRunner
) -> dict[str, Any]:
  start = time.time()

  # Build AF3 folding_input.Input with empty MSA + no templates.
  chains: list[folding_input.ProteinChain] = []
  for c in req.chains:
    chains.append(
        folding_input.ProteinChain(
            id=c.id.upper(),
            sequence=c.sequence,
            ptms=[],
            unpaired_msa="",
            paired_msa="",
            templates=[],
        )
    )

  fold_in = folding_input.Input(name=req.name, chains=chains, rng_seeds=[req.seed])

  ccd = chemical_components.cached_ccd(user_ccd=fold_in.user_ccd)
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_in,
      buckets=cfg.buckets,
      ccd=ccd,
      verbose=False,
      ref_max_modified_date=cfg.max_template_date,
      conformer_max_iterations=cfg.conformer_max_iterations,
      resolve_msa_overlaps=cfg.resolve_msa_overlaps,
  )
  batch = featurised_examples[0]

  rng_key = jax.random.PRNGKey(req.seed)
  result = model_runner.run_inference(batch, rng_key)
  inference_results = model_runner.extract_inference_results(
      batch=batch, result=result, target_name=fold_in.name
  )

  # Pick best sample by ranking_score (same criterion as run_alphafold.py output).
  best = max(inference_results, key=lambda r: float(r.metadata["ranking_score"]))

  # Save outputs.
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  job_id = f"{fold_in.sanitised_name()}_{uuid.uuid4().hex[:8]}"
  mmcif_path = cfg.output_dir / f"{job_id}.cif"
  mmcif_text = best.predicted_structure.to_mmcif()
  mmcif_path.write_text(mmcif_text)

  # Also write a small JSON sidecar with metrics (handy for RL logging/debugging).
  metrics = {
      "job_id": job_id,
      "seed": int(req.seed),
      "ranking_score": float(best.metadata["ranking_score"]),
      "iptm": float(best.metadata["iptm"]),
      "ptm": float(best.metadata["ptm"]),
      "ptm_iptm_average": float(best.metadata["ptm_iptm_average"]),
      "fraction_disordered": float(best.metadata["fraction_disordered"]),
      "has_clash": bool(best.metadata["has_clash"]),
      "plddt_mean": _mean_plddt_from_structure(best.predicted_structure),
      "mmcif_path": str(mmcif_path),
      "latency_s": time.time() - start,
  }
  (cfg.output_dir / f"{job_id}.metrics.json").write_text(json.dumps(metrics, indent=2))

  reward_val = float(best.metadata[req.reward])
  resp: dict[str, Any] = {
      "job_id": job_id,
      "reward_name": req.reward,
      "reward": reward_val,
      **metrics,
  }
  if req.return_mmcif_text:
    resp["mmcif"] = mmcif_text
  return resp


@app.get("/health")
def health() -> dict[str, Any]:
  return {
      "status": "ok",
      "queue_size": _queue.qsize(),
      "model_loaded": bool(_model_runners),
      "num_gpu_workers": len(_model_runners),
      "config": {
          "num_recycles": getattr(_server_cfg, "num_recycles", None),
          "num_diffusion_samples": getattr(_server_cfg, "num_diffusion_samples", None),
          "flash_attention_implementation": getattr(
              _server_cfg, "flash_attention_implementation", None
          ),
          "output_dir": str(getattr(_server_cfg, "output_dir", "")) if _server_cfg else None,
          "server_num_gpus": getattr(_server_cfg, "server_num_gpus", None),
      },
  }


@app.post("/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
  # Enqueue request, wait for worker result.
  loop = asyncio.get_running_loop()
  fut: asyncio.Future = loop.create_future()
  await _queue.put((req, fut))
  return await fut

# TODO(predict_batch):
# Add a `/predict_batch` endpoint for RL rollouts that generate many samples at
# once. Proposed design:
#   - Request: { "items": [ PredictRequest, ... ] } OR { "seeds": [...], "chains": [...] }
#   - Behavior: enqueue items individually (preserve ordering), process with the
#     same single-worker queue to keep GPU memory stable.
#   - Response: list of per-item outputs, including `mmcif_path` for each item.
# This keeps HTTP overhead low while maintaining the current queue semantics.


@app.on_event("startup")
async def _startup():
  global _model_runners
  global _server_cfg

  # `absl_app.run(main)` sets `_server_cfg` before uvicorn starts the app.
  if _server_cfg is None:
    raise RuntimeError(
        "Server config not initialized. Start the server via `python af3_server.py ...` "
        "so flags are parsed and config is set before FastAPI startup."
    )

  cfg = _server_cfg
  gpu_devices = jax.local_devices(backend="gpu")

  if cfg.server_num_gpus is not None:
    if not gpu_devices:
      raise ValueError("--server_num_gpus was set but no GPU devices were found.")
    if cfg.server_num_gpus > len(gpu_devices):
      raise ValueError(
          f"--server_num_gpus={cfg.server_num_gpus} but only {len(gpu_devices)} GPU(s) are visible. "
          "Use CUDA_VISIBLE_DEVICES to control which GPUs are visible."
      )
    selected_devices = list(gpu_devices[: cfg.server_num_gpus])
  else:
    # Backward-compatible: single-device serving using --gpu_device.
    selected_devices = [_choose_device(cfg.gpu_device)]

  _model_runners = []
  for worker_idx, device in enumerate(selected_devices):
    _validate_device_and_attention_impl(device, cfg.flash_attention_implementation)
    runner = ra.ModelRunner(
        config=ra.make_model_config(
            flash_attention_implementation=cfg.flash_attention_implementation,  # type: ignore[arg-type]
            num_diffusion_samples=cfg.num_diffusion_samples,
            num_recycles=cfg.num_recycles,
            return_embeddings=ra._SAVE_EMBEDDINGS.value,
            return_distogram=ra._SAVE_DISTOGRAM.value,
        ),
        device=device,
        model_dir=cfg.model_dir,
    )
    # Load model parameters now (avoid first-request hiccup).
    _ = runner.model_params
    _model_runners.append(runner)
    asyncio.create_task(_worker_loop(worker_idx, runner))

def _build_server_config_from_flags() -> _ServerConfig:
  # Buckets are stored as strings in the flag (same as run_alphafold.py).
  buckets = tuple(int(b) for b in ra._BUCKETS.value)
  max_template_date = datetime.date.fromisoformat(ra._MAX_TEMPLATE_DATE.value)

  # The server reuses `--output_dir` as its output root (it writes *.cif and *.metrics.json).
  output_dir = pathlib.Path(ra._OUTPUT_DIR.value)

  # Mirror run_alphafold.py behavior: if output_dir exists and is non-empty and
  # --force_output_dir is false, create a timestamped directory.
  if (
      not ra._FORCE_OUTPUT_DIR.value
      and output_dir.exists()
      and any(output_dir.iterdir())
  ):
    output_dir = pathlib.Path(
        f"{output_dir}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

  return _ServerConfig(
      model_dir=pathlib.Path(ra.MODEL_DIR.value),
      output_dir=output_dir,
      buckets=buckets,
      max_template_date=max_template_date,
      conformer_max_iterations=ra._CONFORMER_MAX_ITERATIONS.value,
      resolve_msa_overlaps=ra._RESOLVE_MSA_OVERLAPS.value,
      flash_attention_implementation=ra._FLASH_ATTENTION_IMPLEMENTATION.value,
      num_diffusion_samples=ra._NUM_DIFFUSION_SAMPLES.value,
      num_recycles=ra._NUM_RECYCLES.value,
      gpu_device=ra._GPU_DEVICE.value,
      server_num_gpus=_SERVER_NUM_GPUS.value,
  )


def main(_):
  global _server_cfg

  # Server expects inference on requests, not offline JSON inputs.
  if not ra._RUN_INFERENCE.value:
    raise ValueError("af3_server.py requires --run_inference=true.")
  # Keep `run_data_pipeline` flag for parity, but this server currently does not
  # implement the data pipeline (empty MSA + no templates).
  # (We intentionally do not warn by default because run_alphafold.py defaults
  # run_data_pipeline to true. This server currently ignores it.)

  if ra._JAX_COMPILATION_CACHE_DIR.value is not None:
    jax.config.update(
        "jax_compilation_cache_dir", ra._JAX_COMPILATION_CACHE_DIR.value
    )

  if _WORKERS.value != 1:
    raise ValueError(
        "--workers must be 1 for af3_server.py (multiple workers would load the model multiple times)."
    )

  # Make sure output dir exists early.
  if ra._OUTPUT_DIR.value is None:
    raise ValueError("--output_dir must be set.")
  os.makedirs(ra._OUTPUT_DIR.value, exist_ok=True)

  _server_cfg = _build_server_config_from_flags()

  import uvicorn  # pylint: disable=import-outside-toplevel

  # Use the app object directly to avoid re-importing this module (which would
  # re-register absl flags and crash).
  uvicorn.run(app, host=_HOST.value, port=_PORT.value, log_level="info")


if __name__ == "__main__":
  flags.mark_flags_as_required(["output_dir"])
  absl_app.run(main)

