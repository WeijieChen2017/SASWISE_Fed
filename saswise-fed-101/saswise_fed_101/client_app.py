"""saswise-fed-101: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from saswise_fed_101.task import get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        
        # Evaluate on validation set after training
        val_loss, val_accuracy = test(self.net, self.valloader, self.device)
        
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            },
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        
        # By default, evaluate on validation set for regular federated evaluation
        if config.get("eval_on", "val") == "test":
            # If specifically asked to evaluate on test set
            loss, accuracy = test(self.net, self.testloader, self.device)
            return loss, len(self.testloader.dataset), {"accuracy": accuracy, "dataset": "test"}
        else:
            # Regular evaluation on validation set
            loss, accuracy = test(self.net, self.valloader, self.device)
            return loss, len(self.valloader.dataset), {"accuracy": accuracy, "dataset": "val"}


def client_fn(context: Context):
    # Load model and data
    from saswise_fed_101.simulation import ResNetCIFAR10, USE_RESNET
    
    # Select model based on configuration
    if USE_RESNET:
        net = ResNetCIFAR10()
    else:
        from saswise_fed_101.task import Net
        net = Net()
        
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config.get("num-partitions", 10)
    batch_size = context.run_config.get("batch-size", 32)
    local_epochs = context.run_config.get("local-epochs", 10)
    
    trainloader, valloader, testloader = load_data(partition_id, num_partitions, batch_size=batch_size)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, testloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
