# minitorch.py
import torch
import matplotlib.pyplot as plt

class Net:
    def __init__(self): 
        self.layers = []
        self.training = True
        
    def add(self, layer): 
        self.layers.append(layer)
        
    def train(self): 
        self.training = True
        for l in self.layers:
            if hasattr(l, 'train'): l.train()
        return self
        
    def eval(self):  
        self.training = False
        for l in self.layers:
            if hasattr(l, 'eval'): l.eval()
        return self
        
    def forward(self, X):
        for layer in self.layers: 
            X = layer.forward(X)
        return X
        
    def backward(self, dZ):
        for layer in reversed(self.layers): 
            dZ = layer.backward(dZ)
        return dZ
        
    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"): 
                layer.update(lr)

class Linear:
    def __init__(self, nin, nout, device="cpu"):
        # He initialization is better for ReLU networks
        self.W = torch.randn(nin, nout, device=device) * torch.sqrt(torch.tensor(2.0/nin))
        self.b = torch.zeros(nout, device=device)
        self.device = device

    def forward(self, X):
        self.X = X
        return torch.matmul(X, self.W) + self.b

    def backward(self, dZ):
        self.dW = torch.matmul(self.X.t(), dZ)
        self.db = torch.sum(dZ, dim=0)
        return torch.matmul(dZ, self.W.t())

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class ReLU:
    def forward(self, Z):
        self.Z = Z
        return torch.clamp(Z, min=0)

    def backward(self, dA):
        dZ = dA.clone()
        dZ[self.Z <= 0] = 0
        return dZ

class BatchNorm1D:
    def __init__(self, n_features, eps=1e-5, momentum=0.1, device="cpu"):
        self.eps, self.momentum, self.device = eps, momentum, device
        self.gamma = torch.ones(n_features, device=device)
        self.beta = torch.zeros(n_features, device=device)
        self.running_mean = torch.zeros(n_features, device=device)
        self.running_var = torch.ones(n_features, device=device)
        self.training = True

    def train(self): self.training = True; return self
    def eval(self): self.training = False; return self

    def forward(self, X):
        if self.training:
            self.batch_mean = X.mean(dim=0)
            self.batch_var = X.var(dim=0, unbiased=False)
            self.std = torch.sqrt(self.batch_var + self.eps)
            self.X_hat = (X - self.batch_mean) / self.std
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.batch_var
        else:
            self.std = torch.sqrt(self.running_var + self.eps)
            self.X_hat = (X - self.running_mean) / self.std
        
        self.X = X
        return self.gamma * self.X_hat + self.beta

    def backward(self, dY):
        m = dY.size(0)
        self.dbeta = dY.sum(dim=0)
        self.dgamma = (dY * self.X_hat).sum(dim=0)
        dx_hat = dY * self.gamma
        x_mu = self.X - self.batch_mean
        invstd = 1.0 / self.std
        dvar = torch.sum(dx_hat * x_mu * -0.5 * (invstd**3), dim=0)
        dmean = torch.sum(-dx_hat * invstd, dim=0) + dvar * torch.mean(-2.0 * x_mu, dim=0)
        return dx_hat * invstd + (2.0/m) * x_mu * dvar + dmean/m

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

class Dropout:
    def __init__(self, p=0.5, device="cpu"):
        self.p, self.device, self.training = p, device, True

    def train(self): self.training = True; return self
    def eval(self): self.training = False; return self

    def forward(self, X):
        if self.training and self.p > 0:
            keep_prob = 1 - self.p
            self.mask = (torch.rand(X.shape, device=self.device) < keep_prob).float() / keep_prob
            return X * self.mask
        self.mask = torch.ones_like(X)
        return X

    def backward(self, dY):
        return dY * self.mask

class CrossEntropyFromLogits:
    def forward(self, Z, Y):
        self.Y = Y
        Z_stable = Z - Z.max(dim=1, keepdim=True).values
        expZ = torch.exp(Z_stable)
        self.A = expZ / expZ.sum(dim=1, keepdim=True)
        log_probs = Z_stable - torch.log(expZ.sum(dim=1, keepdim=True))
        correct_log_probs = log_probs[torch.arange(Z.shape[0]), Y]
        return -correct_log_probs.mean()

    def backward(self, n_classes):
        m = self.Y.shape[0]
        Y_one_hot = torch.zeros(m, n_classes, device=self.A.device)
        Y_one_hot[torch.arange(m), self.Y] = 1.0
        return (self.A - Y_one_hot) / m
    

def train_model(net, train_loader, val_loader, criterion, epochs=10, lr=0.1):
    history = {
        "train_loss": [], "train_acc": [], 
        "val_loss": [], "val_acc": []
    }
    batch_losses = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        
        for X, y in train_loader:
            X = X.view(X.shape[0], -1).to(device)
            y = y.to(device)
            
            outputs = net.forward(X)
            loss = criterion.forward(outputs, y)
            batch_losses.append(loss.item())
            
            dZ = criterion.backward(n_classes=10)
            net.backward(dZ)
            net.update(lr)
            
            epoch_loss += loss.item()
            
            # --- Added: Calculate Training Accuracy ---
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)
            
        # Validation
        net.eval()
        correct_val, total_val, val_loss = 0, 0, 0
        for X, y in val_loader:
            X = X.view(X.shape[0], -1).to(device)
            y = y.to(device)
            outputs = net.forward(X)
            val_loss += criterion.forward(outputs, y).item()
            preds = torch.argmax(outputs, dim=1)
            correct_val += (preds == y).sum().item()
            total_val += y.size(0)
            
        # Store all metrics
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["train_acc"].append(correct_train / total_train)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(correct_val / total_val)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {history['train_acc'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}")
        
    return history, batch_losses

def plot_history(history, title=""):
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot Loss
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], 'r-o', label="train")
    plt.plot(epochs, history["val_loss"], 'orange', linestyle='--', label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_acc"], 'b-s', label="train")
    plt.plot(epochs, history["val_acc"], 'cyan', linestyle='--', label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_per_batch_loss(batch_losses, title="Per-batch loss"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(batch_losses, color='gray', alpha=0.4, label='Batch Loss')
    
    # Add trend line
    if len(batch_losses) > 50:
        import numpy as np
        window = 50
        weights = np.ones(window) / window
        trend = np.convolve(batch_losses, weights, mode='valid')
        plt.plot(range(window-1, len(batch_losses)), trend, 'r', label='Trend')

    plt.xlabel('Batch Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()