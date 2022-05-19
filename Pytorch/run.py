from Data import dataset
import torch
from model import NeuralNetwork
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt


larning_rate = 1e-3
batch_size = 32
load_shape = 256
target_shape = 256
loss_fn = nn.CrossEntropyLoss()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'using {device} device')
model = NeuralNetwork().to(device)
optim = torch.optim.RMSprop(model.parameters(),lr=larning_rate)
transform = transforms.Compose([
    transforms.Resize((load_shape,load_shape)),
    transforms.RandomCrop((target_shape, target_shape)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

data_train = dataset('/content/training', transform=transform)
data_val = dataset('/content/validation', transform=transform)

X = DataLoader(data_train, batch_size=batch_size, shuffle=True)
Y = DataLoader(data_val, batch_size=batch_size, shuffle=True)
print(model)

def calculate_accuracy(y_pred, y):
  correct = (y_pred.argmax(1).to(device) == y.argmax(1).to(device)).type(torch.float).sum()
  acc = correct / y.shape[0]
  return acc

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred= model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

Train_acc, val_acc = list(), list()
Train_loss, val_loss = list(), list()

EPOCHS = 50

best_valid_loss = float('inf')

for epoch in range(EPOCHS):

    start_time = time.monotonic()

    train_loss, train_acc = train(model, X, optim, loss_fn, device)
    valid_loss, valid_acc = evaluate(model, Y, loss_fn, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'ABOLFAZL.pt')

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    Train_acc.append(train_acc*100)
    val_acc.append(valid_acc*100)
    Train_loss.append(train_loss)
    val_loss.append(valid_loss)

x = [i+1 for i in range(EPOCHS)]
plt.plot(np.array(x), np.array(Train_acc), label = 'train')
plt.plot(np.array(x), np.array(val_acc), label = 'val')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.plot(np.array(x), np.array(Train_loss), label = 'train')
plt.plot(np.array(x), np.array(val_loss), label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')