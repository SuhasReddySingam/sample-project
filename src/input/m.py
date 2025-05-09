import os
import sys
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url, extract_gz
from rdkit import Chem
import numpy as np
from .feature import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Data
from .vocab import *

# Constants
max_len = 128
atom_dict = {
    5: 'C', 6: 'C', 9: 'O', 12: 'N', 15: 'N', 21: 'F', 23: 'S', 25: 'Cl',
    26: 'S', 28: 'O', 34: 'Br', 36: 'P', 37: 'I', 39: 'Na', 40: 'B', 
    41: 'Si', 42: 'Se', 44: 'K'
}

class SmilesAlphabet:
    def __init__(self):
        chars = b'#%)(+-.1032547698=ACBEDGFIHKMLONPSRUTWVY[Z]_acbedgfihmlonsruty'
        self.chars = np.frombuffer(chars, dtype='uint8')
        self.size = len(self.chars)
        self.encoding = np.zeros(256, dtype='uint8') + 255  # Default to missing
        self.encoding[self.chars] = np.arange(self.size)

    def encode(self, s):
        s = np.frombuffer(s, dtype='uint8')
        return self.encoding[s]

smilebet = SmilesAlphabet()

class MoleculeEmbedder:
    def __init__(self, vocab_path='data/smiles_vocab.pkl'):
        """Initialize with path to pre-trained SMILES vocabulary."""
        self.drug_vocab = WordVocab.load_vocab(vocab_path)
        self.max_len = max_len

    def embed_molecule(self, smiles):
        """Embed a single SMILES string into a torch_geometric Data object."""
        # Create a Data object
        data = Data()

        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            mol.UpdatePropertyCache(strict=False)

        # Atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)
        if len(x) > self.max_len:
            x = x[:self.max_len]
        data.x = torch.from_numpy(x).to(torch.int64)
        data.graph_len = len(data.x)

        # Bonds
        edges_list=[]
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i >= self.max_len or j >= self.max_len:
                continue
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))  # Add reverse edge
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)
        data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
        data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)

        # Store original SMILES
        data.smiles_ori = smiles

        # Tokenize SMILES using drug_vocab
        content = []
        flag = 0
        sm = smiles
        while flag < len(sm):
            if flag + 1 < len(sm) and self.drug_vocab.stoi.get(sm[flag:flag + 2]):
                content.append(self.drug_vocab.stoi.get(sm[flag:flag + 2]))
                flag += 2
            else:
                content.append(self.drug_vocab.stoi.get(sm[flag], self.drug_vocab.unk_index))
                flag += 1

        if len(content) > self.max_len:
            content = content[:self.max_len]
        data.smile_len = len(content)

        # Mask for padding
        mask = torch.ones(self.max_len)
        mask[len(content):] = 0
        data.mask = mask

        # Pad SMILES embedding
        X = content
        if self.max_len > len(X):
            padding = [self.drug_vocab.pad_index] * (self.max_len - len(X))
            X.extend(padding)
        smile_emb = torch.tensor(X, dtype=torch.long)
        data.smile_emb = smile_emb

        # Atom positions in SMILES (for validation)
        tem = [i for i, c in enumerate(X) if c in atom_dict]
        data.atom_len = tem

        # Encode SMILES with SmilesAlphabet
        smiles_f = smiles.encode('utf-8').upper()
        smiles_f = torch.from_numpy(smilebet.encode(smiles_f)).long()
        data.smiles_f = smiles_f

        # Validation (optional)
        if len(tem) != data.x.size(0):
            print("Warning: Atom count mismatch!")
            print(f"SMILES atoms: {len(tem)}, Graph atoms: {data.x.size(0)}")
            print(f"SMILES embedding: {smile_emb}")

        return data

# Example usage
if __name__ == "__main__":
    # Initialize the embedder
    embedder = MoleculeEmbedder(vocab_path='data/smiles_vocab.pkl')

    # Example SMILES string
    smiles = "CCO"  # Ethanol
    try:
        # Embed the molecule
        embedded_data = embedder.embed_molecule(smiles)
        
        # Print some results
        print(f"Atom features (x): {embedded_data.x}")
        print(f"Edge index: {embedded_data.edge_index}")
        print(f"Edge attributes: {embedded_data.edge_attr}")
        print(f"SMILES embedding: {embedded_data.smile_emb}")
        print(f"Mask: {embedded_data.mask}")
        print(f"Original SMILES: {embedded_data.smiles_ori}")
    except Exception as e:
        print(f"Error embedding molecule: {e}")