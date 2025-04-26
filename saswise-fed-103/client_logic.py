import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from flwr.client import Client, NumPyClient
from collections import OrderedDict
from datetime import datetime
import numpy as np

from models import ResNet20, set_weights, get_weights

# Try to import log from utils2, otherwise create a basic logger
try:
    from utils2 import log
except ImportError:
    from logging import INFO, ERROR
    # Simple print logger if utils2.log is not available
    def log(level, msg, *args):
        if level >= INFO: # Basic filtering
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {level}: {msg % args}")

class FlowerClient(NumPyClient):
    def __init__(
        self, 
        cid, 
        trainset, 
        testset, 
        device, 
        epochs=10, 
        batch_size=128, 
        learning_rate=0.01, 
        momentum=0.9,
        client_metrics_ref=None,
        log_dir=None
    ):
        self.cid = cid
        self.trainset = trainset
        self.testset = testset
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.client_metrics_ref = client_metrics_ref or {}
        self.log_dir = log_dir
        
        # Create model
        self.model = ResNet20().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
    
    def get_parameters(self, config):
        return get_weights(self.model)
    
    def set_parameters(self, parameters):
        set_weights(self.model, parameters)
        return None
    
    def fit(self, parameters, config):
        # Record start time
        start_time = time.time()
        
        # Get round number from config
        current_round = config.get("round", 0)
        log(INFO, f"[Client {self.cid}] Fitting in round {current_round}")
        
        # Update local model with global parameters
        self.set_parameters(parameters)
        
        # Create data loader
        trainloader = DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True
        )
        
        # Create optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )
        
        # Train
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for epoch in range(self.epochs):
            batch_loss = 0.0
            batch_correct = 0
            batch_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_total += targets.size(0)
                batch_correct += predicted.eq(targets).sum().item()
            
            # Log epoch results
            epoch_loss = batch_loss / (batch_idx + 1)
            epoch_acc = batch_correct / batch_total * 100
            log(INFO, f"[Client {self.cid}] Epoch {epoch+1}/{self.epochs}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            
            # Update metrics
            train_loss += batch_loss
            train_correct += batch_correct
            train_total += batch_total
        
        # Calculate final metrics
        final_train_loss = train_loss / (self.epochs * len(trainloader))
        final_train_acc = train_correct / train_total * 100
        
        # Store metrics in shared dictionary
        if self.client_metrics_ref is not None:
            if current_round not in self.client_metrics_ref:
                self.client_metrics_ref[current_round] = {}
            
            if "clients" not in self.client_metrics_ref[current_round]:
                self.client_metrics_ref[current_round]["clients"] = {}
            
            self.client_metrics_ref[current_round]["clients"][self.cid] = {
                "train_loss": float(final_train_loss),
                "train_accuracy": float(final_train_acc),
                "train_size": train_total
            }
        
        # Save metrics to file if log_dir is provided
        if self.log_dir:
            metrics_dir = os.path.join(self.log_dir, f"client_{self.cid}")
            os.makedirs(metrics_dir, exist_ok=True)
            
            metrics_file = os.path.join(metrics_dir, f"round_{current_round}.json")
            metrics = {
                "train_loss": float(final_train_loss),
                "train_accuracy": float(final_train_acc),
                "train_size": train_total,
                "round": current_round,
                "client_id": self.cid,
                "timestamp": datetime.now().isoformat()
            }
            
            import json
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Calculate fit duration
        fit_time = time.time() - start_time
        
        # Return updated parameters and metrics
        return self.get_parameters({}), train_total, {
            "train_loss": final_train_loss,
            "train_accuracy": final_train_acc,
            "fit_duration": fit_time
        }
    
    def evaluate(self, parameters, config):
        # Record start time
        start_time = time.time()
        
        # Get round number from config
        current_round = config.get("round", 0)
        log(INFO, f"[Client {self.cid}] Evaluating in round {current_round}")
        
        # Update local model with global parameters
        self.set_parameters(parameters)
        
        # Create data loader
        testloader = DataLoader(
            self.testset, batch_size=self.batch_size
        )
        
        # Evaluate
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        # Calculate final metrics
        final_test_loss = test_loss / len(testloader)
        final_test_acc = test_correct / test_total * 100
        
        # Store metrics in shared dictionary
        if self.client_metrics_ref is not None:
            if current_round not in self.client_metrics_ref:
                self.client_metrics_ref[current_round] = {}
            
            if "clients" not in self.client_metrics_ref[current_round]:
                self.client_metrics_ref[current_round]["clients"] = {}
            
            # Update client metrics with evaluation results
            if self.cid in self.client_metrics_ref[current_round]["clients"]:
                self.client_metrics_ref[current_round]["clients"][self.cid].update({
                    "test_loss": float(final_test_loss),
                    "test_accuracy": float(final_test_acc),
                    "test_size": test_total
                })
            else:
                self.client_metrics_ref[current_round]["clients"][self.cid] = {
                    "test_loss": float(final_test_loss),
                    "test_accuracy": float(final_test_acc),
                    "test_size": test_total
                }
        
        # Calculate evaluation duration
        eval_time = time.time() - start_time
        
        # Return loss, num_examples, and metrics
        return float(final_test_loss), test_total, {
            "test_accuracy": float(final_test_acc),
            "eval_duration": eval_time
        }

def client_fn_factory(train_sets, testset, device, training_params, log_dir, client_metrics_ref):
    """Factory function that creates a function which instantiates FlowerClient instances."""
    
    def client_fn(cid: str):
        """Client builder function that instantiates a FlowerClient with id `cid`."""
        # Convert client ID to integer
        client_id = int(cid)
        
        # Get client's training set
        trainset = train_sets[client_id]
        
        # Create and return FlowerClient
        return FlowerClient(
            cid=client_id,
            trainset=trainset,
            testset=testset,
            device=device,
            epochs=training_params.get("epochs", 10),
            batch_size=training_params.get("batch_size", 128),
            learning_rate=training_params.get("learning_rate", 0.01),
            momentum=training_params.get("momentum", 0.9),
            client_metrics_ref=client_metrics_ref,
            log_dir=log_dir
        )
    
    return client_fn 