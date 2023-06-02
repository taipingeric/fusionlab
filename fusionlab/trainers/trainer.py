import torch
from tqdm.auto import tqdm
import numpy as np

# ref: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback

class Trainer:
    def __init__(self, device):
        self.device = device

    def train_step(self, data):
        data = self._data_to_device(data)
        inputs, target = data
        pred = self.model(inputs)
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    def val_step(self, data):
        data = self._data_to_device(data)
        inputs, target = data
        with torch.no_grad():
            pred = self.model(inputs)
            loss = self.loss_fn(pred, target)
        return loss.item()

    def train_epoch(self):
        self.model.train()
        epoch_loss = []
        for _, data in enumerate(tqdm(self.train_dataloader, leave=False)):
            batch_loss = self.train_step(data)
            epoch_loss.append(batch_loss)
        return np.mean(epoch_loss)
    
    def val_epoch(self):
        self.model.eval()
        epoch_loss = []
        for _, data in enumerate(tqdm(self.val_dataloader, leave=False)):
            batch_loss = self.val_step(data)
            epoch_loss.append(batch_loss)
        return np.mean(epoch_loss)
    
    def on_fit_begin(self):
        pass
    def on_fit_end(self):
        pass

    def on_epoch_begin(self):
        pass
    def on_epoch_end(self):
        pass

    def _data_to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: v.to(self.device) for k, v in data.items()}
        elif isinstance(data, list):
            return [v.to(self.device) for v in data]
        else:
            raise NotImplementedError

    def fit(self, model, train_dataloader, val_dataloader, epochs, optimizer, loss_fn):
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_log = {'loss': []}
        self.val_log = {'loss': []}

        self.on_fit_begin()
        for epoch in tqdm(range(epochs)):
            self.on_epoch_begin()
            train_epoch_loss = self.train_epoch()
            self.train_log['loss'].append(train_epoch_loss)

            if self.val_dataloader: 
                val_epoch_loss = self.val_epoch()
                self.val_log['loss'].append(val_epoch_loss)
            
            print(f'''[{epoch}/{epochs}] train_loss: {self.train_log['loss'][-1]:.4f} \
    val_loss: {self.val_log['loss'][-1]:.4f}''')
            self.on_epoch_end()
        self.on_fit_end()
        return 

if __name__ == "__main__":
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 3, 3)
            self.pool = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
            )
            self.cls = torch.nn.Linear(3, 10)
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = self.cls(x)
            return x
    
    from abc import ABC, abstractmethod
    class Metric(ABC):
        def __init__(self):
            pass

        @abstractmethod
        def reset():
            raise NotImplementedError("reset method is not implemented!")

        @abstractmethod
        def update():
            raise NotImplementedError("update method is not implemented!")
        
        @abstractmethod
        def compute():
            raise NotImplementedError("compute method is not implemented!")
    
    # class Accuracy(Metric):
        

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # mnist
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader
    train_dataset = MNIST(root='data/', train=True, transform=ToTensor(), download=True)
    val_dataset = MNIST(root='data/', train=False, transform=ToTensor(), download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = FakeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(device)
    trainer.fit(model, 
                train_dataloader, 
                val_dataloader, 
                10, 
                optimizer, 
                loss_fn)
    
    
    