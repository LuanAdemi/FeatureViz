from torchvision import datasets, transforms
import torch
from model import CNN
import numpy as np

device = torch.device("cpu")

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = CNN(1, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

num_train = len(train_loader.dataset)
num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

for epoch in range(10):
    model.train()

    loss_train = []
    pred_train = []

    for i, (element, label) in enumerate(train_loader, 1):

        element = element.to(device)
        label = label.to(device)

        output = model(element)
        pred = torch.softmax(output, dim=1).max(dim=1, keepdim=True)[1]

        optimizer.zero_grad()

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        loss_train.append(loss.item())
        pred_train.append(((pred == label.view_as(pred)).type(torch.float)).mean(dim=0).item())

        print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f ACC: %.4f' %
              (epoch, i, num_batch_train, np.mean(loss_train), 100 * np.mean(pred_train)))

torch.save(model.state_dict(), "model.pth")
