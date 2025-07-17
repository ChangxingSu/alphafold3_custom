# Scripts for AlphaFold 3 Data Preparation

This directory contains helper scripts for preparing input data for AlphaFold 3.

---


## `create_af3_input_from_a3ms.py`

### Purpose

This script is a utility to construct an AlphaFold 3 input JSON file from pre-existing `.a3m` files. It is useful if you have already generated your MSAs through a different process and simply need to package them into the required JSON format.

The script groups `.a3m` files by a common protein name, parses them, and injects their content into a template JSON file.

### Logic for MSA Pairing

The script distinguishes between paired and unpaired MSAs based on a simple rule:
- If a sequence's header line in the `.a3m` file contains `tax=`, it is treated as a **paired MSA** entry.
- Otherwise, it is treated as an **unpaired MSA** entry.

### Usage

```bash
python scripts/create_af3_input_from_a3ms.py \
    --a3m_dir /path/to/your/a3m_files/ \
    --template_path /path/to/template.json \
    --output_dir /path/to/your/output_jsons/
```

**Arguments:**

*   `--a3m_dir`: Path to the directory containing your `.a3m` files. **Files must be named in the format `{name}_{chainId}.a3m`** (e.g., `1a8h_A.a3m`, `1a8h_B.a3m`).
*   `--template_path`: Path to a template JSON file that will be used as a base for the new inputs.
*   `--output_dir`: Directory where the final, formatted JSON files will be saved. A new file will be created for each unique protein name found.

```