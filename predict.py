import argparse
import torch
import numpy as np
import yaml
from src.dataset import GenomicDataset
from src.transformer_model import MetaGenoTransformer

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
        label_path="data/processed/labels.npy",  # still required by constructor but won't be loaded
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
        for x, _ in dataloader:
            output = model(x)
            predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    np.save("logs/predictions.npy", predictions)
    print("Predictions saved to logs/predictions.npy")

if __name__ == '__main__':
    predict()
