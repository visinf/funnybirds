import torch.nn as nn
from abc import abstractmethod

class ModelExplainerWrapper:

    def __init__(self, model, explainer):
        """
        A generic wrapper that takes any model and any explainer to putput model predictions 
        and explanations that highlight important input image part.
        Args:
            model: PyTorch neural network model
            explainer: PyTorch model explainer    
        """
        self.model = model
        self.explainer = explainer

    def predict(self, input):
        return self.model.forward(input)

    def explain(self, input):
        return self.explainer.explain(self.model, input)


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

class StandardModel(AbstractModel):
    """
    A wrapper for standard PyTorch models (e.g. ResNet, VGG, AlexNet, ...).
    Args:
        model: PyTorch neural network model
    """

    def forward(self, input):
        return self.model(input)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

import torch

class ViTModel(AbstractModel):
    """
    A wrapper for ViT models.
    Args:
        model: PyTorch neural network model
    """

    def forward(self, input):
        input = torch.nn.functional.interpolate(input, (224,224)) # ViT expects input of size 224x224

        return self.model(input)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


from models.bcos.data_transforms import AddInverse
from torch.autograd import Variable

class BcosModel(AbstractModel):
    """
    A wrapper for Bcos models.
    Args:
        model: PyTorch bcos model
    """

    def forward(self, input):
        _input = Variable(AddInverse()(input), requires_grad=True)
        return self.model(_input)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

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