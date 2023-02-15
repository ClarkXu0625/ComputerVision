import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from utils import AverageMeter


class evaluation:
    def __init__(self, model):
        self._model = model
        

    def eval(self, test_loader: DataLoader) -> float:
        """ Model evaluation, return the model accuracy over test set """

        self._model.eval()
        correct_predictions = 0 # number of corrected prediction
        evaluation_num = 0  # number of samples being evaluated

        with torch.no_grad():
            for image, label in test_loader:
                output = self._model(images)
                _, predicted_class = torch.max(output.data, 1)

                # Increment correct predictions (if prediction is correct) and evaluations
                correct_predictions += (predicted_class == label).sum().item()
                evaluation_num += label.size(0)
        
        return  correct_predictions/evaluation_num