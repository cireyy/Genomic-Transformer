import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd


class GenomicDataset(Dataset):
    def __init__(self, data_dir, label_path=None, family_history_path=None, transform=None, predict=False):
        """
        Args:
            data_dir (str): Directory containing 22 chromosome CSV files named chr1.csv to chr22.csv
            label_path (str): Path to labels (optional if predict=True)
            family_history_path: Path to family history (optional)
            predict (bool): If True, skip loading labels (used during inference)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.predict = predict

        # Load all 22 chromosome files into a list of arrays
        self.chromosome_data = [
            pd.read_csv(os.path.join(data_dir, f"chr{i}.csv"), header=0).iloc[:, 1:].values
            for i in range(1, 23)
        ]

        self.num_samples = self.chromosome_data[0].shape[0]
        assert all(chr_data.shape[0] == self.num_samples for chr_data in self.chromosome_data), "Mismatch in sample count across chromosomes"

        # Load labels unless in predict mode
        if not self.predict:
            self.labels = self._load_file(label_path)
            assert len(self.labels) == self.num_samples, "Mismatch between data and label sample count"
        else:
            self.labels = None

        # Load family history if provided
        self.family_history = None
        if family_history_path:
            self.family_history = self._load_file(family_history_path)
            assert self.family_history.shape[0] == self.num_samples, "Mismatch in family history and sample count"

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
        # Pad SNPs for all chromosomes for this sample
        max_snps = max([chr_data.shape[1] for chr_data in self.chromosome_data])
        sample_padded = [
            np.pad(chr_data[idx], (0, max_snps - chr_data.shape[1]), mode='constant', constant_values=0)
            for chr_data in self.chromosome_data
        ]
        sample = np.stack(sample_padded, axis=0)  # shape: [22, max_snps]

        if self.family_history is not None:
            family_history = torch.FloatTensor(self.family_history[idx])
        else:
            family_history = torch.zeros(6)

        if self.transform:
            sample = self.transform(sample)

        sample = torch.LongTensor(sample)

        if self.predict:
            return sample, family_history
        else:
            label = torch.FloatTensor(self.labels[idx])
            return sample, label, family_history
