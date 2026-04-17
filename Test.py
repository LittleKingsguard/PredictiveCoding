import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# import copy 

import predictive_coding as pc

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using {device}')

# load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
train_dataset = datasets.MNIST('./data', download=True, train=True, transform=transform)
test_dataset = datasets.MNIST('./data', download=True, train=False, transform=transform)

batch_size = 500
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f'# train images: {len(train_dataset)} and # test images: {len(test_dataset)}')

""" Defining a model

A model can be initalised in the same way as pytorch model, with the addition of pc.PCLayer() to include latent variables in the model.

A PCLayer() contains the activities of a layer of latent variables under pclayer._x. A PCLayer() also contains the energy associated with that activity under pclayer._energy which is computed with 0.5 *(inputs['mu'] - inputs['x'])**2 where inputs['x'] is the activity of that layer and inputs['mu'] is the input to that layer.

Check out the PCLayer() class in predictive_coding/pc_layer.py for more information. """

input_size = 28*28  # 28x28 images
hidden_size = 256
output_size = 10    # 10 classes
activation_fn = nn.ReLU
loss_fn = lambda output, _target: 0.5 * (output - _target).pow(2).sum() # this loss function holds to the error of the output layer of the model


model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    pc.PCLayer(),
    activation_fn(),
    nn.Linear(hidden_size, hidden_size),
    pc.PCLayer(),
    activation_fn(),
    nn.Linear(hidden_size, output_size)
)
model.train()   # set the model to training mode
model.to(device)

"""### Defining a model trainer
The predictive coding library is based around a `pc.PCTrainer()`. 

This trainer orchestrate the activity and parameter updates of the model to minimise the total error of the model. The total error is given by the sum of the energies in each pclayer as well as the loss functions. """

# number of inference iterations where the latent states x are updated. Inference does not run till convergence but for a fixed number of iterations
T = 20                              

# options for the update of the latent state x
optimizer_x_fn = optim.SGD          # optimizer for latent state x, SGD perform gradient descent. Other alternative are Adam, RMSprop, etc. 
optimizer_x_kwargs = {'lr': 0.01}   # optimizer parameters for latent state x to pass to the optimizer. The best learning rate will depend on the task and the optimiser. 
                                    # Other parameters such as momentum, weight_decay could also be set here with additional elements, e.g., "momentum": 0.9, "weight_decay": 0.01

# options for the update of the parameters p
update_p_at = 'last'                # update parameters p at the last iteration, can be set to 'all' to implement ipc (https://arxiv.org/abs/2212.00720)
optimizer_p_fn = optim.Adam         # optimizer for parameters p
optimizer_p_kwargs = {'lr': 0.001}  # optimizer parameters for parameters p, 0.001 is a good starting point for Adam, but it should be adjusted for the task

trainer = pc.PCTrainer(model, 
    T = T, 
    optimizer_x_fn = optimizer_x_fn,
    optimizer_x_kwargs = optimizer_x_kwargs,
    update_p_at = update_p_at,   
    optimizer_p_fn = optimizer_p_fn,
    optimizer_p_kwargs = optimizer_p_kwargs,
)

# get classification accuracy of the model
def test(model, dataset, batch_size=1000):
    model.eval()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        pred = model(data)
        _, predicted = torch.max(pred, -1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    model.train()
    return round(correct / total, 4)

    """Train the model

trainer.train_on_batch() is called for each batch of data. This function updates the activity of the latent states and the parameters for the given batch of data. """

epochs = 3

test_acc = np.zeros(epochs + 1)
test_acc[0] = test(model, test_dataset)
batch_acc = []
batch_acc.append(test_acc[0] * 100)
for epoch in range(epochs):
    # Initialize the tqdm progress bar
    with tqdm(train_loader, desc=f'Epoch {epoch+1} - Test accuracy: {test_acc[epoch]:.3f}') as pbar:
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            # convert labels to one-hot encoding
            label = F.one_hot(label, num_classes=output_size).float()
            trainer.train_on_batch(
                inputs=data,
                loss_fn=loss_fn,
                loss_fn_kwargs={'_target': label}
            )
            batch_acc.append(test(model, test_dataset) * 100)    
    test_acc[epoch + 1] = test(model, test_dataset)
    pbar.set_description(f'Epoch {epoch + 1} - Test accuracy: {test_acc[epoch + 1]:.3f}')

plt.plot(test_acc)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.show()

#The trained model achieves a classification accuracy of above 95% on MNIST which is comparable to a backpropagation trained model with the same architecture.

# Below is reference model using same layers and sizes using a backpropagation System


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Initialize the model
BPmodel = SimpleNN()
print(BPmodel)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(BPmodel.parameters(), lr=0.01, momentum=0.9)

#Define new trainer
BPtrain_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
BPmodel.to(device)

# Training loop
num_epochs = 3
train_losses = []
train_accuracy = []
BP_batch_accuracy = []
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

BPmodel.eval()
correct = 0
total = 0
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = BPmodel(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
train_accuracy.append(accuracy)
BP_batch_accuracy.append(accuracy)
print(f'Accuracy on the test set: {accuracy:.2f}%')


for epoch in range(num_epochs):
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = BPmodel(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

        BPmodel.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = BPmodel(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        BP_batch_accuracy.append(accuracy)
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            train_losses.append(running_loss/100)
            running_loss = 0.0
    # Evaluate the model
    BPmodel.eval()
    correct = 0
    total = 0
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = BPmodel(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    train_accuracy.append(accuracy)
    print(f'Accuracy on the test set: {accuracy:.2f}%')

print('Training finished!')

# Evaluate the model
BPmodel.eval()
correct = 0
total = 0
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f'Accuracy on the test set: {accuracy:.2f}%')


axis = np.arange(0, 361)
plt.plot(axis, BP_batch_accuracy, label="Backpropagation Model")
plt.plot(axis, batch_acc, label="Predictive Coding Model")
plt.xlabel('Train Events x500')
plt.ylabel('Test Accuracy')
plt.legend()
plt.savefig("Test Result1.png")
print(batch_acc)