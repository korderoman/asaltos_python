import torch.nn as nn
from torchvision.models import ResNet18_Weights
import torchvision.models as models
class Extractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Cargar modelo ResNet preentrenado y modificar para extraer características
        self.weights = ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=self.weights)
        self.resnet.fc = nn.Identity()  # Eliminar la capa de clasificación para obtener características
        self.resnet.to(device).eval()
        self.model = self.resnet
    def forward(self, x):
        return self.model(x)