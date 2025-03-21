import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, log_loss
import numpy as np
import os


class Trainer:
    def __init__(self, model, dataset, config):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self.model = model.to(self.device)
        self.dataset = dataset
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.best_model_path = config['best_model_path']
        self.patience = config['patience']

    def train_fold(self, train_idx, val_idx):
        train_subset = Subset(self.dataset, train_idx)
        val_subset = Subset(self.dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.config['batch_size'], shuffle=False)

        best_val_auc = 0.0
        patience_counter = 0
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                x, y, family_history = batch
                x, y, family_history = x.to(self.device), y.to(self.device), family_history.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x, family_history)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            metrics = self.evaluate(val_loader)
            print(
                f"Epoch {epoch + 1}/{self.config['num_epochs']}, Loss: {total_loss:.4f}, Val AUC: {metrics['AUROC']:.4f}, "
                f"F1: {metrics['F1']:.4f}, Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, LogLoss: {metrics['LogLoss']:.4f}")

            if metrics['AUROC'] > best_val_auc:
                best_val_auc = metrics['AUROC']
                os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.best_model_path)
                print("Model saved with best validation AUC!")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

    def evaluate(self, dataloader):
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                x, y, family_history = batch
                x, y, family_history = x.to(self.device), y.to(self.device), family_history.to(self.device)
                logits = self.model(x, family_history)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_labels.append(y.cpu().numpy())
                all_preds.append(probs)

        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)

        aucs = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])]
        f1s = [f1_score(all_labels[:, i], (all_preds[:, i] > 0.5).astype(int), zero_division=1) for i in
               range(all_labels.shape[1])]
        precisions = [precision_score(all_labels[:, i], (all_preds[:, i] > 0.5).astype(int), zero_division=1) for i in
                      range(all_labels.shape[1])]
        recalls = [recall_score(all_labels[:, i], (all_preds[:, i] > 0.5).astype(int), zero_division=1) for i in
                   range(all_labels.shape[1])]
        log_losses = [log_loss(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])]

        return {
            "AUROC": np.mean(aucs),
            "F1": np.mean(f1s),
            "Precision": np.mean(precisions),
            "Recall": np.mean(recalls),
            "LogLoss": np.mean(log_losses)
        }

    def train(self):
        print("Starting 5-Fold Cross Validation...")
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(np.arange(len(self.dataset)))):
            print(f"Fold {fold + 1}/5")
            self.train_fold(train_idx, val_idx)
        print("Training complete!")