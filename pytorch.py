# -*- coding: utf-8 -*-
"""pytorch.ipynb

Original file is located at
    https://colab.research.google.com/drive/117jQQqKdotQJD5Lb6E74URdPIG1hicZa
"""

from torch import nn
from torch.optim import Adam
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

train = datasets.MNIST(root='data',download =True,train=True,transform=ToTensor())
dataset = DataLoader(train,32)

class ImageClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(1,32,(3,3)),
        nn.ReLU(),
        nn.Conv2d(32,64,(3,3)),
        nn.ReLU(),
        nn.Conv2d(64,64,(3,3)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*(28-6)*(28-6),10)
    )

  def forward(self,x):
    return self.model(x)

#instances,loss, opt

clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(),lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

import matplotlib.pyplot as plt

# training

loss_history = []  # List to store loss values

if __name__ == '__main__':
  for epoch in range(10):
    for batch in dataset:
      X,y = batch
      X,y = X.to('cuda'), y.to('cuda')
      yhat = clf(X)
      loss = loss_fn(yhat,y)

      opt.zero_grad()
      loss.backward()
      opt.step()

    print(f"Epoch: {epoch} loss is {loss.item()}")
    loss_history.append(loss.item()) # Store the loss after each epoch

# Visualize the loss
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

import torch

# Save the model
torch.save(clf.state_dict(), 'image_classifier.pt')
