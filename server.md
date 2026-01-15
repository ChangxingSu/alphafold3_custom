## Running AlphaFold 3 as a Server (RL/Reward Loop)

This repo includes a minimal queued inference server in `af3_server.py` designed
for RL/reward-loop integration:

*   **Persistent model**: model parameters are loaded once at server startup.
*   **Fixed inference config**: `num_recycles` and `num_diffusion_samples` are
    fixed at startup (recommended to avoid recompilations and latency jitter).
*   **Server-side queue**: requests are queued and processed sequentially by a
    single worker (GPU-friendly).
*   **Structure output**: each request saves an mmCIF file and a small metrics
    JSON sidecar.

### Install server dependencies

The server requires additional Python deps:

```
pip install fastapi uvicorn pydantic
```

### Configure (command line flags; aligned with `run_alphafold.py`)

This server accepts the same core flags as `run_alphafold.py` for consistency.
For RL usage, the important ones are:

*   `--model_dir`: path to model parameters directory (defaults to `~/models`)
*   `--output_dir`: where to write `*.cif` and `*.metrics.json`
*   `--num_recycles`: fixed recycle count
*   `--num_diffusion_samples`: fixed diffusion sample count
*   `--flash_attention_implementation`: `triton` / `cudnn` / `xla`
*   `--gpu_device`: GPU index
*   `--jax_compilation_cache_dir`: directory for JAX compilation cache (recommended)

For GPU compute capability 7.x you must also set:

*   `XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter`
*   `--flash_attention_implementation=xla`

### Start the server

**Important**: use a single worker to avoid multiple model loads and GPU memory
issues.

```
python af3_server.py \
  --model_dir=/root/models \
  --output_dir=/root/af_server_output \
  --num_recycles=3 \
  --num_diffusion_samples=1 \
  --flash_attention_implementation=triton \
  --gpu_device=0 \
  --jax_compilation_cache_dir=/root/jax_cache \
  --host 0.0.0.0 --port 8000 --workers 1
```

### Make a request (antibody–antigen complex, empty MSA/no templates)

Request format is a list of chains (`H/L/A/...`), plus an RNG seed:

Note: currently the server exposes a single-item endpoint (`/predict`). A
`/predict_batch` endpoint is planned (see TODO in `af3_server.py`) but not yet
implemented.

```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ab_ag",
    "seed": 0,
    "reward": "iptm",
    "chains": [
      {"id": "H", "sequence": "EVQLQQSGAELARPGASVKMSCKASGYTFTSYTMHWVKQRPGQGLEWIGYINPSSGYSNYNQKFKDKATLTADKSSSTAYMQLSSLTSEDSAVYYCSRPVVRLGYNFDYWGQGSTLTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVP"},
      {"id": "L", "sequence": "EIVLTQSPAITAASLGQKVTITCSASSSVSYMHWYQQKSGTSPKPWIYEISKLASGVPARFSGSGSGTSYSLTISSMEAEDAAIYYCQQWNYPFTFGSGTKLEIKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRN"},
      {"id": "A", "sequence": "VHQAISPRTLNAWVKVVEEKAFSPEVIPMFSALSEGATPQDLNTMLNTVGGHQAAMQMLKETINEEAAEWDRVHPVHAGPIAPGQMREPRGSDIAGTTSTLQEQIGWMTNNPPIPVGEIYKRWIILGLNKIVRMYSPTSILDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPAATLEEMMTACQG"}
    ],
    "return_mmcif_text": false
  }'
```

The response includes:

*   `iptm`, `ptm`, `ranking_score`, `plddt_mean`
*   `mmcif_path` (saved structure), plus a `*.metrics.json` sidecar next to it