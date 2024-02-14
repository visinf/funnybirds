import torch.nn as nn
from abc import abstractmethod

class AbstractModel(nn.Module):
    def __init__(self, model):
        """
        An abstract wrapper for PyTorch models implementing functions required for evaluation.
        Args:
            model: PyTorch neural network model
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, input):
        return self.model

class ProtoPNetModel(AbstractModel):
    """
    A wrapper for ProtoPNet models.
    Args:
        model: PyTorch ProtoPNet model
    """
    def __init__(self, model, load_model_dir, epoch_number_str):
        super().__init__(model)
        self.model = model
        self.load_model_dir = load_model_dir
        self.epoch_number_str = epoch_number_str

    def forward(self, input, return_min_distances = False):
        logits, min_distances = self.model(input)
        if not return_min_distances:
            return logits
        else:
            return logits, min_distances

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)