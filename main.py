import argparse
import torch
from torch.utils.data import DataLoader
import yaml
import os

from src.dataset import GenomicDataset
from src.transformer_model import MetaGenoTransformer
from src.trainer import Trainer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load dataset
    dataset = GenomicDataset(
        data_dir=config['data_dir'],
        label_path=config['label_path'],
        family_history_path=config.get('family_history_path', None)
    )

    # Initialize model
    model = MetaGenoTransformer(
        embedding_dim=512,
        attention_dim=512,
        num_heads=8,
        num_layers=2,
        num_tasks=6,
        dropout=0.1
    )

    # Start training
    trainer = Trainer(model, dataset, config)
    trainer.train()


if __name__ == '__main__':
    main()
