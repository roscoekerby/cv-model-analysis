from abc import ABC, abstractmethod

class EvaluatorBase(ABC):
    def __init__(self, model_path):
        self.model_path = model_path

    @abstractmethod
    def evaluate(self, images, labels):
        pass

    @abstractmethod
    def show_metrics(self, y_true, y_pred):
        pass
