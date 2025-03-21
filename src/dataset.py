import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd


class GenomicDataset(Dataset):
    def __init__(self, data_dir, label_path, family_history_path=None, transform=None):
        """
        Args:
            data_dir (str): Directory containing 22 chromosome CSV files named chr1.csv to chr22.csv
            label_path (str): Path to labels
            family_history_path: Path to family history
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load all 22 chromosome files into a list of DataFrames
        self.chromosome_data = [
            pd.read_csv(os.path.join(data_dir, f"chr{i}.csv"), header=0).iloc[:, 1:].values
            for i in range(1, 23)
        ]

        # Check that all chromosomes have same number of samples
        self.num_samples = self.chromosome_data[0].shape[0]
        assert all(chr_data.shape[0] == self.num_samples for chr_data in self.chromosome_data), "Mismatch in sample count across chromosomes"


        # Load labels
        self.labels = self._load_file(label_path)
        assert len(self.labels) == self.num_samples, "Mismatch between data and label sample count"

        # Load family history if provided
        self.family_history = None
        if family_history_path:
            self.family_history = self._load_file(family_history_path)
            assert len(self.labels) == self.num_samples, "Mismatch between data and label sample count"

    def _load_file(self, path):
        ext = os.path.splitext(path)[1]
        if ext == '.npy':
            return np.load(path)
        elif ext == '.csv':
            return pd.read_csv(path, header=0).values
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Collect SNPs for all chromosomes for this sample
        sample = [chr_data[idx] for chr_data in self.chromosome_data]
        max_snps = max([chr_data.shape[1] for chr_data in self.chromosome_data])

        sample_padded = [np.pad(chr_data[idx], (0, max_snps - len(chr_data[idx])), mode='constant', constant_values=0)
                         for chr_data in self.chromosome_data]
        sample = np.stack(sample_padded, axis=0)  # shape: [22, max_snps]

        label = self.labels[idx]  # shape: [6]

        if self.family_history is not None:
            family_history = self.family_history[idx]  # shape: [6]
        else:
            family_history = np.zeros(6)

        if self.transform:
            sample = self.transform(sample)

        return torch.LongTensor(sample), torch.FloatTensor(label), torch.FloatTensor(family_history)