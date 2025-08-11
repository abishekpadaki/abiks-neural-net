The sekeleton below covers like 90% of what you'll need for most PyTorch projects. The pattern is always the same:

Model: __init__ for layers, forward for data flow
Data: Raw → tensors → datasets → dataloaders
Training: Forward → loss → backward → step, repeat
A few things I always forget and have to look up:

model.train() vs model.eval() - don't skip these!
optimizer.zero_grad() before backward pass
torch.no_grad() during validation to save memory
Moving data to device with .to(device)

The skeleton handles batching, validation, learning rate scheduling, and model saving. You can strip out what you don't need or add more layers/complexity to the model class.
For classification, just swap nn.MSELoss() for nn.CrossEntropyLoss() and maybe add a softmax or sigmoid activation to your final layer.
This structure scales from simple linear regression all the way up to transformers - it's the same fundamental pattern everywhere in PyTorch.


---

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 1. MODEL DEFINITION
# =============================================================================

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super().__init__()
        
        # Define layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Define forward pass
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.output(x)  # No activation on final layer for regression
        return x

# =============================================================================
# 2. DATA PREPARATION 
# =============================================================================

def prepare_data(X_train, y_train, X_val, y_val, batch_size=32):
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# =============================================================================
# 3. TRAINING SETUP
# =============================================================================

def setup_training(model, learning_rate=0.001):
    # Loss function
    criterion = nn.MSELoss()  # or nn.CrossEntropyLoss() for classification
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    return criterion, optimizer, scheduler

# =============================================================================
# 4. TRAINING LOOP
# =============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=100, device='cpu'):
    
    model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # Update scheduler
        scheduler.step()
        
        # Record losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] | '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses, val_losses

# =============================================================================
# 5. USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Create dummy data
    X_train = torch.randn(1000, 10)
    y_train = torch.randn(1000, 1)
    X_val = torch.randn(200, 10)
    y_val = torch.randn(200, 1)
    
    # Setup
    model = MyModel(input_size=10, hidden_size=64, output_size=1)
    train_loader, val_loader = prepare_data(X_train, y_train, X_val, y_val)
    criterion, optimizer, scheduler = setup_training(model)
    
    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=50, device=device
    )
    
    # Save model
    torch.save(model.state_dict(), 'model.pth')
    
    # Load model later
    # model.load_state_dict(torch.load('model.pth'))
    # model.eval()
```


## QUICK REFERENCE CHECKLIST:

1. Define model class inheriting from nn.Module
2. Implement __init__ (define layers) and forward (define flow)
3. Prepare data → tensors → datasets → dataloaders
4. Choose criterion, optimizer, scheduler
5. Training loop: train mode → forward → loss → backward → step
6. Validation loop: eval mode → no_grad → forward → loss (no backward)
7. Save/load model state_dict
