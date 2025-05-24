# evaluators/pytorch_evaluator.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .evaluator_base import EvaluatorBase


class PyTorchEvaluator(EvaluatorBase):
    def __init__(self, model_path):
        super().__init__(model_path)
        # Handle different PyTorch save formats
        checkpoint = torch.load(model_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            # If it's a checkpoint dictionary, try to extract the model
            if 'model' in checkpoint:
                self.model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # You'll need to reconstruct the model architecture here
                # This is a limitation - we need the model class definition
                raise ValueError(
                    "Checkpoint contains state_dict but no model architecture. Please save the full model or provide model architecture.")
            elif 'model_state_dict' in checkpoint:
                raise ValueError(
                    "Checkpoint contains model_state_dict but no model architecture. Please save the full model or provide model architecture.")
            else:
                # Assume the dict itself is the state dict
                raise ValueError("Cannot determine model from checkpoint dictionary. Please save the full model.")
        else:
            # Direct model object
            self.model = checkpoint

        self.model.eval()

    def evaluate(self, images, labels):
        y_true, y_pred = [], []
        with torch.no_grad():
            for img, lbl in zip(images, labels):
                if isinstance(img, torch.Tensor):
                    img = img.unsqueeze(0)  # Add batch dimension
                else:
                    img = torch.tensor(img).unsqueeze(0)

                output = self.model(img)
                pred = torch.argmax(output, dim=1)

                if isinstance(lbl, torch.Tensor):
                    y_true.append(lbl.item())
                else:
                    y_true.append(int(lbl))
                y_pred.append(pred.item())
        return y_true, y_pred

    def show_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Accuracy: {acc:.2f}")
        print(f"Balanced Accuracy: {bacc:.2f}")

        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()