# models/onnx_evaluator.py
import onnxruntime as ort
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .evaluator_base import EvaluatorBase

class ONNXEvaluator(EvaluatorBase):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.session = ort.InferenceSession(model_path)

    def evaluate(self, images, labels):
        y_true, y_pred = [], []
        input_name = self.session.get_inputs()[0].name

        for img, lbl in zip(images, labels):
            img_np = img.numpy().astype(np.float32)[None, :]
            outputs = self.session.run(None, {input_name: img_np})
            pred = np.argmax(outputs[0], axis=1)[0]
            y_true.append(lbl.item())
            y_pred.append(pred)
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