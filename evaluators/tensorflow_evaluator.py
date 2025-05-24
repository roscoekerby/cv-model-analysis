# models/tf_evaluator.py
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .evaluator_base import EvaluatorBase

class TFEvaluator(EvaluatorBase):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.model = load_model(model_path)

    def evaluate(self, images, labels):
        y_pred = np.argmax(self.model.predict(images), axis=1)
        y_true = labels.numpy()
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
