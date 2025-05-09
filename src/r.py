import os
import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import subprocess
import fm  # RNA-FM module
from datetime import datetime
import sys

class RNA_Processor:
    def __init__(self, RNA_type="user", spot_rna2d_dir="src/SPOT-RNA-2D"):
        """
        Initialize RNA_Processor for single-sequence processing.

        Args:
            input_sequence (str): RNA sequence (A, U, G, C, T).
            RNA_type (str): Identifier for the sequence type (default: "user").
            spot_rna2d_dir (str): Path to SPOT-RNA-2D directory (default: correct path).
        """
        self.RNA_type = RNA_type
        self.spot_rna2d_dir = os.path.abspath(spot_rna2d_dir)  # Normalize to absolute path


        # Create directories for FASTA files and SPOT-RNA-2D outputs
        self.fasta_dir = os.path.abspath(os.path.join(os.getcwd(), "fasta_files"))
        self.output_dir = os.path.abspath(os.path.join(os.getcwd(), "spot_rna_outputs"))
        os.makedirs(self.fasta_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self,input_sequence):
        """Process the input sequence and return a Data object."""
        # Truncate sequence to 500 (SPOT-RNA-2D limit)
        self.input_sequence = input_sequence.upper()
        if not isinstance(input_sequence, str) or not all(c in 'AUGCT' for c in self.input_sequence):
            raise ValueError("Input sequence must be a string containing only A, U, G, C, or T")
        sequence = self.input_sequence[:500]
        entry_id = "user_sequence"
        target_id = "user_target"

        # Generate unique filename suffix using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{entry_id}_{timestamp}"

        # Generate contact map using SPOT-RNA-2D
        contact_file, fasta_file = self._generate_contact_map(sequence, unique_id)
        if contact_file is None or fasta_file is None:
            raise RuntimeError("Failed to generate contact map with SPOT-RNA-2D")

        # Load and process contact map
        matrix = np.loadtxt(contact_file)
        matrix[matrix < 0.5] = 0
        matrix[matrix >= 0.5] = 1

        # Generate RNA-FM embeddings
        rna_emb = self._generate_rna_fm_embeddings(sequence, target_id)
        if rna_emb is None:
            raise RuntimeError("Failed to generate RNA-FM embeddings")

        # Create graph data
        one_hot_sequence = [self.char_to_one_hot(char) for char in sequence]
        edges = np.argwhere(matrix == 1)

        x = torch.tensor(one_hot_sequence, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        y = torch.tensor([0.0], dtype=torch.float32)  # Placeholder for pKd
        rna_len = x.size()[0]

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            t_id=target_id,
            e_id=unique_id,
            emb=rna_emb,
            rna_len=rna_len,
        )

        return data

    def _generate_contact_map(self, sequence, unique_id):
        """Generate contact map using SPOT-RNA-2D."""
        try:
            # Create FASTA file without extension
            fasta_file = os.path.join(self.fasta_dir, f"{unique_id}")
            with open(fasta_file, 'w') as f:
                f.write(f">{unique_id}\n{sequence}\n")
            print(f"Created FASTA file at: {fasta_file}")
            if not os.path.exists(fasta_file):
                raise FileNotFoundError(f"FASTA file {fasta_file} was not created")

            # Verify SPOT-RNA-2D script
            spot_rna2d_script = os.path.join(self.spot_rna2d_dir, "run.py")
            if not os.path.exists(spot_rna2d_script):
                raise FileNotFoundError(
                    f"SPOT-RNA-2D script not found at {spot_rna2d_script}. "
                    f"Please ensure SPOT-RNA-2D is installed at {self.spot_rna2d_dir}."
                )

            # Run SPOT-RNA-2D
            cmd = [
                sys.executable,  # Use current Python executable
                os.path.abspath(spot_rna2d_script),
                "--rna_id", unique_id,
                "--input_feats", os.path.abspath(self.fasta_dir),
                "--outputs", os.path.abspath(self.output_dir),
                "--single_seq", "1"
            ]
            print(f"Running SPOT-RNA-2D with command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=self.spot_rna2d_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"SPOT-RNA-2D stdout: {result.stdout}")
            if result.returncode != 0:
                print(f"SPOT-RNA-2D error: {result.stderr}")
                return None, None

            # Check for output file
            contact_file = os.path.join(self.output_dir, f"{unique_id}.prob_single")
            if not os.path.exists(contact_file):
                print(f"SPOT-RNA-2D did not generate {contact_file}")
                return None, None

            return contact_file, fasta_file

        except subprocess.CalledProcessError as e:
            print(f"Subprocess error running SPOT-RNA-2D: {e.stderr}")
            return None, None
        except Exception as e:
            print(f"Error generating contact map: {str(e)}")
            return None, None

    def _generate_rna_fm_embeddings(self, sequence, target_id):
        """Generate embeddings using RNA-FM."""
        try:
            print("Loading RNA-FM model...")
            model, alphabet = fm.pretrained.rna_fm_t12()
            model.eval()
            batch_converter = alphabet.get_batch_converter()

            data = [(target_id, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[12])
                token_embeddings = results["representations"][12]  # Shape: [1, seq_len, 640]

            return token_embeddings.squeeze(0)  # Shape: [seq_len, 640]

        except Exception as e:
            print(f"Error generating RNA-FM embeddings: {str(e)}")
            return None

    def char_to_one_hot(self, char):
        """Convert nucleotide to one-hot vector."""
        char = char.upper()
        if char == 'T':
            char = 'U'
        mapping = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
        return mapping.get(char, [0, 0, 0, 0])  # Default for unknown characters