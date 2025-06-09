import argparse
import torch
import numpy as np
import yaml
from src.dataset import GenomicDataset
from src.transformer_model import MetaGenoTransformer
import os
import pandas as pd


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    # Load dataset
    dataset = GenomicDataset(
        data_dir="data/processed",
        label_path="data/processed/labels.npy",
        family_history_path="data/processed/family_history.npy",
        predict=True
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.get("batch_size", 64), shuffle=False)

    # Load model
    model = MetaGenoTransformer(
        embedding_dim=512,
        attention_dim=512,
        num_heads=8,
        num_layers=2,
        num_tasks=6,
        dropout=0.1
    )
    model.load_state_dict(torch.load(config['best_model_path']))
    model.eval()

    predictions = []

    with torch.no_grad():
       for batch in dataloader:
            x, family_history = batch
            output = model(x, family_history)
            predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    family_history_path = "data/processed/family_history.csv"
    sample_ids = pd.read_csv(family_history_path).iloc[:, 0].values

    log_dir = config.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Save .npy
    np.save(os.path.join(log_dir, "predictions.npy"), predictions)
    print(f"Predictions saved to {log_dir}/predictions.npy")

    # Save .csv
    df = pd.DataFrame(predictions, columns=["AF", "HT", "HCL", "T2D", "CAD", "IS"])
    df.insert(0, "Sample_ID", sample_ids)
    df.to_csv(os.path.join(log_dir, "predictions.csv"), index=False)
    print(f"Predictions saved to {log_dir}/predictions.csv")

if __name__ == '__main__':
    predict()
