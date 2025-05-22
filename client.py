import torch
import flwr as fl
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
from medmnist import INFO, Evaluator
from train_all_datasets import create_resnet50, evaluate
from resnet3d import resnet3d50

class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        data_flag: str,
        train_loader,
        val_loader,
        test_loader,
        device,
        num_local_epochs: int = 5,
        is_3d: bool = False
    ):
        self.client_id = client_id
        self.data_flag = data_flag
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_local_epochs = num_local_epochs
        self.is_3d = is_3d
        
        # Create model
        self._setup_model()
        
    def _setup_model(self):
        """Initialize the model based on dataset type."""
        # Get dataset info
        info = INFO[self.data_flag]
        self.task = info['task']
        self.n_channels = info['n_channels']
        self.n_classes = len(info['label'])
        
        # Create appropriate model
        if not self.is_3d:
            self.model = create_resnet50(in_channels=self.n_channels, num_classes=self.n_classes)
        else:
            self.model = resnet3d50(num_classes=self.n_classes, in_channels=self.n_channels)
        
        self.model.to(self.device)
        
        # Setup loss function
        if self.task == "multi-label, binary-class":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    
    def get_parameters(self, config) -> List[np.ndarray]:
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        # Get training config
        local_epochs = config.get("local_epochs", self.num_local_epochs)
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Train model
        self.model.train()
        for epoch in range(local_epochs):
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in self.train_loader:
                # Move data to device
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device)
                
                # For 3D datasets, normalize
                if self.is_3d:
                    inputs = (inputs - 0.5) / 0.5
                
                # Forward + backward + optimize
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                if self.task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = self.criterion(outputs, targets)
                else:
                    targets = targets.squeeze().long()
                    loss = self.criterion(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
                
                # Track stats
                train_loss += loss.item()
                
                if self.task != 'multi-label, binary-class':
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            # Calculate training accuracy for this epoch
            if self.task != 'multi-label, binary-class' and total > 0:
                accuracy = 100.0 * correct / total
                print(f"Client {self.client_id}, Epoch {epoch+1}: Train Acc: {accuracy:.2f}%")
        
        # Compute number of samples used for training
        num_train_samples = len(self.train_loader.dataset)
        
        # Evaluate model after training
        eval_metrics = self.evaluate(parameters, config)
        
        # Return updated parameters and training metrics
        return self.get_parameters(config), num_train_samples, eval_metrics
    
    def evaluate(self, parameters, config):
        """Evaluate the model on the local test set."""
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate on test set
        test_true, test_score = evaluate(
            self.model, 
            self.test_loader, 
            self.task, 
            self.device, 
            is_3d=self.is_3d
        )
        
        # Calculate metrics
        evaluator = Evaluator(self.data_flag, 'test')
        test_metrics = evaluator.evaluate(test_score)
        test_auc, test_acc = test_metrics
        
        # Number of examples in test set
        num_test_samples = len(self.test_loader.dataset)
        
        # Return evaluation metrics
        return {
            "acc": float(test_acc),
            "auc": float(test_auc),
        }, num_test_samples 