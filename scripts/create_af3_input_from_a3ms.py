
import os
import json
import argparse
from collections import defaultdict
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_a3m(a3m_path: str) -> tuple[str, str, str]:
    """
    Parses an .a3m file to extract the query sequence, paired MSA, and unpaired MSA.

    Args:
        a3m_path: Path to the .a3m file.

    Returns:
        A tuple containing:
        - The query sequence (the first sequence in the file).
        - A string of all paired MSA entries (headers contain 'tax=').
        - A string of all unpaired MSA entries (headers do not contain 'tax=').
    """
    try:
        with open(a3m_path, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        logging.error(f"Could not read file {a3m_path}: {e}")
        return "", "", ""

    if not lines or not lines[0].startswith('>'):
        logging.warning(f"File {a3m_path} is empty or does not start with a header.")
        return "", "", ""

    query_sequence = lines[1].strip() if len(lines) > 1 else ""
    if not query_sequence:
        logging.warning(f"Could not extract query sequence from {a3m_path}.")
        return "", "", ""

    paired_msa_entries = []
    unpaired_msa_entries = []

    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            header = lines[i]
            if i + 1 < len(lines):
                sequence = lines[i+1]
                entry = header + sequence
                # Check for 'tax=' to decide if the MSA is paired
                if 'tax=' in header.lower():
                    paired_msa_entries.append(entry)
                else:
                    unpaired_msa_entries.append(entry)
                i += 2  # Move to the next entry
            else:
                i += 1 # End of file, dangling header
        else:
            i += 1 # Should not happen in a valid a3m

    return query_sequence, "".join(paired_msa_entries), "".join(unpaired_msa_entries)


def main():
    """
    Main function to discover, process, and generate AlphaFold 3 input files.
    """
    parser = argparse.ArgumentParser(
        description="Generate AlphaFold 3 input JSON from a directory of .a3m files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--a3m_dir",
        type=str,
        required=True,
        help="Directory containing .a3m files named in the format '{name}_{chainId}.a3m'."
    )
    parser.add_argument(
        "--template_path",
        type=str,
        required=True,
        help="Path to the template JSON file (e.g., 'input.json')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the output JSON files will be saved."
    )
    args = parser.parse_args()

    # 1. Discover and group .a3m files by protein name
    a3m_groups = defaultdict(list)
    logging.info(f"Scanning for .a3m files in '{args.a3m_dir}'...")
    for filename in os.listdir(args.a3m_dir):
        if filename.endswith(".a3m"):
            parts = filename.rsplit('.', 1)[0].split('_', 1)
            if len(parts) == 2:
                protein_name, chain_id = parts
                full_path = os.path.join(args.a3m_dir, filename)
                a3m_groups[protein_name].append({'path': full_path, 'chain_id': chain_id})
            else:
                logging.warning(f"Skipping malformed filename (expected 'name_chain.a3m'): {filename}")

    if not a3m_groups:
        logging.error("No valid .a3m files found. Exiting.")
        return

    # 2. Load the JSON template
    try:
        with open(args.template_path, 'r') as f:
            template = json.load(f)
        logging.info(f"Successfully loaded template from '{args.template_path}'.")
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load or parse template JSON: {e}")
        return

    # 3. Process each group of files
    os.makedirs(args.output_dir, exist_ok=True)
    for name, chain_files in a3m_groups.items():
        logging.info(f"Processing protein '{name}' with {len(chain_files)} chain(s)...")
        
        # Create a fresh copy of the template for this protein
        output_data = json.loads(json.dumps(template))
        output_data['name'] = name
        output_data['sequences'] = []

        # Sort chains by ID for consistent ordering
        sorted_chains = sorted(chain_files, key=lambda x: x['chain_id'])

        for chain_info in sorted_chains:
            path = chain_info['path']
            chain_id = chain_info['chain_id'].upper()
            
            logging.info(f"  - Parsing Chain {chain_id} from {os.path.basename(path)}...")
            query_seq, paired_msa, unpaired_msa = parse_a3m(path)

            if not query_seq:
                logging.warning(f"    Skipping chain {chain_id} due to missing query sequence.")
                continue

            # Construct the protein entry for the sequence list
            protein_entry = {
                "protein": {
                    "id": chain_id,
                    "sequence": query_seq,
                    "pairedMsa": paired_msa,
                    "unpairedMsa": unpaired_msa,
                    "templates": [] # Add empty templates list for compatibility
                }
            }
            output_data['sequences'].append(protein_entry)

        # 4. Write the final JSON file for the protein
        if output_data['sequences']:
            output_path = os.path.join(args.output_dir, f"{name}.json")
            try:
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=4)
                logging.info(f"Successfully created output file: '{output_path}'")
            except IOError as e:
                logging.error(f"Failed to write output file {output_path}: {e}")
        else:
            logging.warning(f"No valid chains were processed for protein '{name}', so no output file was generated.")

if __name__ == "__main__":
    main()
