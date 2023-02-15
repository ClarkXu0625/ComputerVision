def infer(self, sample: Tensor) -> int:
    """ Model inference: input an image, return its class index """
    self._model.eval()    
    inference = self._model(sample)[0]

    # Get the class with the highest probability
    predicted_class = torch.argmax(inference).item()

    return predicted_class