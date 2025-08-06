#!/usr/bin/env python3
"""
npz_to_af3_integrated.py

Integrated script: Generate AlphaFold 3 JSON input files directly from NPZ files
Supports TSV filtering, large-scale file processing, and memory optimization

Usage:
    python npz_to_af3_integrated.py --npz_dir /path/to/npz --template template.json --output_dir output/
    python npz_to_af3_integrated.py --npz_dir /path/to/npz --template template.json --output_dir output/ --tsv_file proteins.tsv
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional

# Handle boltz imports gracefully
try:
    from boltz.data import const
    from boltz.data.types import MSA
    BOLTZ_AVAILABLE = True
except ImportError:
    print("Warning: boltz library not available. Please install boltz.")
    BOLTZ_AVAILABLE = False
    # Mock classes for basic functionality
    class MSA:
        @classmethod
        def load(cls, path):
            raise NotImplementedError("Boltz library required for NPZ processing")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NPZToAF3Processor:
    """Integrated NPZ to AlphaFold 3 JSON processor"""
    
    def __init__(self, npz_dir: str, template_path: str, output_dir: str, 
                 tsv_file: Optional[str] = None, num_processes: int = None):
        self.npz_dir = Path(npz_dir)
        self.template_path = Path(template_path)
        self.output_dir = Path(output_dir)
        self.tsv_file = Path(tsv_file) if tsv_file else None
        self.num_processes = num_processes or os.cpu_count()
        
        if not BOLTZ_AVAILABLE:
            raise RuntimeError("Boltz library is required for NPZ processing")
        
        # Load template
        self.template = self._load_template()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_template(self) -> Dict:
        """Load AlphaFold 3 JSON template"""
        try:
            with open(self.template_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load template: {e}")
            raise
    
    def _load_protein_names(self) -> Optional[List[str]]:
        """Load protein names list from TSV file"""
        if not self.tsv_file or not self.tsv_file.exists():
            return None
        
        try:
            with open(self.tsv_file, 'r') as f:
                names = [line.strip().split('\t')[0] for line in f if line.strip()]
            logger.info(f"Loaded {len(names)} protein names from TSV")
            logger.info(f"{"ig7l" in names}")
            return names
        except Exception as e:
            logger.warning(f"Failed to read TSV file: {e}")
            return None
    
    def _find_npz_files(self) -> List[Path]:
        """Find NPZ files, supports TSV filtering"""
        target_names = self._load_protein_names()
        
        # Use rglob to support subdirectories
        npz_files = list(self.npz_dir.rglob("*.npz")) if self.npz_dir.is_dir() else [self.npz_dir]
        
        if target_names:
            filtered_files = []
            for npz_file in npz_files:
                base_name = npz_file.stem
                protein_name = base_name.rsplit('_', 1)[0]  # Remove chain suffix
                
                if protein_name in target_names:
                    filtered_files.append(npz_file)
            
            logger.info(f"After filtering: {len(npz_files)} → {len(filtered_files)} files")
            return filtered_files
        
        logger.info(f"Found {len(npz_files)} NPZ files")
        return npz_files
    
    def _npz_to_a3m_string(self, msa: MSA) -> str:
        """Convert MSA object to A3M format string"""
        lines = []
        
        for seq_idx, taxonomy_id, res_start, res_end, del_start, del_end in msa.sequences:
            # Build header
            header = f">seq_{seq_idx}"
            if taxonomy_id != -1:
                header += f" OX={taxonomy_id}"
            lines.append(header)
            
            # Reconstruct sequence (including deletions)
            sequence_str = []
            current_residues = msa.residues[res_start:res_end]
            current_deletions = msa.deletions[del_start:del_end]
            
            deletion_map = {d["res_idx"]: d["deletion"] for d in current_deletions}
            
            for i, res_data in enumerate(current_residues):
                res_type_token = res_data["res_type"]
                amino_acid_letter = const.prot_token_to_letter.get(
                    const.tokens[res_type_token], "X"
                )
                sequence_str.append(amino_acid_letter)
                
                # Add deleted residues (represented by -)
                if i in deletion_map:
                    sequence_str.append("-" * deletion_map[i])
            
            lines.append("".join(sequence_str))
        
        return "\n".join(lines) + "\n"
    
    def _parse_a3m_content(self, a3m_content: str) -> Tuple[str, str, str]:
        """Parse A3M content, extract query sequence, paired MSA, and unpaired MSA"""
        lines = a3m_content.strip().split('\n')
        
        if not lines or not lines[0].startswith('>'):
            raise ValueError("Invalid A3M format")
        
        query_sequence = lines[1].strip() if len(lines) > 1 else ""
        if not query_sequence:
            raise ValueError("Query sequence is empty")
        
        paired_msa_entries = []
        unpaired_msa_entries = []
        
        # Add query sequence to unpaired MSA (first sequence)
        query_header = ">query"
        query_entry = query_header + '\n' + query_sequence + '\n'
        # Add query sequence to paired MSA and unpaired MSA
        unpaired_msa_entries.append(query_entry)
        paired_msa_entries.append(query_entry)

        i = 2  # Start from the second sequence (skip query sequence)
        while i < len(lines):
            if lines[i].startswith('>'):
                header = lines[i]
                if i + 1 < len(lines):
                    sequence = lines[i+1]
                    entry = header + '\n' + sequence + '\n'
                    
                    # Check paired MSA identifiers
                    if 'tax=' in header.lower() or 'ox=' in header.lower():
                        paired_msa_entries.append(entry)
                    else:
                        unpaired_msa_entries.append(entry)
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return query_sequence, "".join(paired_msa_entries), "".join(unpaired_msa_entries)
    
    def _process_single_npz(self, npz_path: Path) -> Tuple[str, str, str, str]:
        """Process a single NPZ file"""
        try:
            msa = MSA.load(npz_path)
            a3m_string = self._npz_to_a3m_string(msa)
            query_seq, paired_msa, unpaired_msa = self._parse_a3m_content(a3m_string)
            
            # Extract chain ID
            base_name = npz_path.stem
            parts = base_name.rsplit('_', 1)
            chain_id = parts[1] if len(parts) == 2 else 'A'
            
            return chain_id.upper(), query_seq, paired_msa, unpaired_msa
        except Exception as e:
            logger.error(f"Failed to process file {npz_path}: {e}")
            return None, None, None, None
    
    def _group_files_by_protein(self, npz_files: List[Path]) -> Dict[str, List[Tuple[str, Path]]]:
        """Group NPZ files by protein name"""
        protein_groups = defaultdict(list)
        
        for npz_file in npz_files:
            base_name = npz_file.stem
            parts = base_name.rsplit('_', 1)
            
            if len(parts) == 2:
                protein_name, chain_id = parts
            else:
                protein_name = base_name
                chain_id = 'A'
            
            protein_groups[protein_name].append((chain_id.upper(), npz_file))
        
        return protein_groups
    
    def process_all(self):
        """Main processing function"""
        npz_files = self._find_npz_files()
        
        if not npz_files:
            logger.error("No NPZ files found")
            return
        
        protein_groups = self._group_files_by_protein(npz_files)
        logger.info(f"Processing {len(protein_groups)} protein groups")
        
        processed_count = 0
        for protein_name, chain_files in protein_groups.items():
            logger.info(f"Processing protein: {protein_name} ({len(chain_files)} chains)")
            
            # Create template copy
            output_data = json.loads(json.dumps(self.template))
            output_data['name'] = protein_name
            output_data['sequences'] = []
            
            # Sort by chain ID
            chain_files.sort(key=lambda x: x[0])
            
            valid_chains = 0
            for chain_id, npz_path in chain_files:
                chain_id_result, query_seq, paired_msa, unpaired_msa = self._process_single_npz(npz_path)
                
                if chain_id_result is None:
                    continue
                
                if not query_seq:
                    logger.warning(f"Skipping chain {chain_id}: query sequence is empty")
                    continue
                
                protein_entry = {
                    "protein": {
                        "id": chain_id,
                        "sequence": query_seq,
                        "pairedMsa": paired_msa,
                        "unpairedMsa": unpaired_msa,
                        "templates": []
                    }
                }
                output_data['sequences'].append(protein_entry)
                valid_chains += 1
            
            # Write output JSON
            if output_data['sequences']:
                output_path = self.output_dir / f"{protein_name}.json"
                try:
                    with open(output_path, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    logger.info(f"Created: {output_path} ({valid_chains} chains)")
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Failed to write file {output_path}: {e}")
        
        logger.info(f"Processing complete! Generated {processed_count} JSON files")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Integrated NPZ to AlphaFold 3 JSON processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --npz_dir ./npz_files --template template.json --output_dir ./output/
  %(prog)s --npz_dir ./npz_files --template template.json --output_dir ./output/ --tsv_file proteins.tsv
  %(prog)s --npz_dir ./npz_files --template template.json --output_dir ./output/ --num_processes 8
        """
    )
    
    parser.add_argument("--npz_dir", required=True, 
                        help="Directory path containing NPZ files")
    parser.add_argument("--template_path", required=True,
                        help="AlphaFold 3 template JSON file path")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for JSON files")
    parser.add_argument("--tsv_file", 
                        help="TSV file containing protein names to process (optional)")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of processes for parallel processing (default: CPU core count)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input paths
    if not Path(args.npz_dir).exists():
        logger.error(f"NPZ directory does not exist: {args.npz_dir}")
        return
    
    if not Path(args.template_path).exists():
        logger.error(f"Template file does not exist: {args.template_path}")
        return
    
    # Create and run processor
    try:
        processor = NPZToAF3Processor(
            npz_dir=args.npz_dir,
            template_path=args.template_path,
            output_dir=args.output_dir,
            tsv_file=args.tsv_file
        )
        
        processor.process_all()
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()