"""
SASWISE Fed-102 Flower Client implementation.
"""

import torch
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

class FlowerClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""
    
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def get_parameters(self, config) -> NDArrays:
        """Get parameters from the network."""
        # Return model parameters as a list of NumPy arrays
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set parameters in the network."""
        # Set model parameters from a list of NumPy arrays
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the network on the local data."""
        # Update local model parameters
        self.set_parameters(parameters)
        
        # Get local training data
        num_examples = len(self.trainloader.dataset)
        
        # Import train function from task
        from saswise_fed_102.task import train
        
        # Train the model using the local data
        loss = train(
            net=self.net,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            device=self.device,
        )
        
        # Return local model parameters and statistics
        return self.get_parameters(config={}), num_examples, {"loss": loss}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the network on the local data."""
        # Update local model parameters
        self.set_parameters(parameters)
        
        # Import test function from task
        from saswise_fed_102.task import test
        
        # Evaluate global model on local data
        results = test(self.net, self.valloader, self.device)
        
        # Return statistics
        return results["loss"], len(self.valloader.dataset), {"accuracy": results["accuracy"]}
    
    def to_client(self) -> fl.client.Client:
        """Convert NumPyClient to Client."""
        return fl.client.NumPyClientWrapper(self)
