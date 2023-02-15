import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

class inference:
    def __init__(self, model):
        self._model = model

    def infer(self, sample: Tensor) -> int:
        """ Model inference: input an image, return its class index """
        self._model.eval()   
        
        with torch.no_grad():
            
            inference = self._model(sample)[0]

            # Get the class with the highest probability
            predicted_class = torch.argmax(inference).item()

            return predicted_class