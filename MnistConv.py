import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import tensorboard

class Conv2dFCModel(nn.Module):
    def __init__(self):
        super(Conv2dFCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64*7*7, 28)
        self.fc2 = nn.Linear(28, 10)
        self.act = nn.ReLU()
    
    def forward(self, x): #x: [B, 1, 28, 28]
        x = self.conv1(x) # --> [B, 32, 28, 28]
        x = self.act(x)
        x = self.pool(x)  # --> [B, 32, 14, 14]
        
        x = self.conv2(x)  # --> [B, 64, 14, 14]
        x = self.act(x)
        x = self.pool(x)  # --> [B, 64, 7, 7]

        x = x.view(-1, 64*7*7)
        x = self.fc1(x)  #(1, 28)
        x = self.act(x)

        x = self.fc2(x) #(1, 10)
        return x



transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformer)
data_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformer)

loader_train = DataLoader(data_train, 64, shuffle=True)
loader_test = DataLoader(data_test, 64, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = Conv2dFCModel().to(device)
optimizer =  optim.AdamW(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    epoc_loss = 0
    for images, labels in loader_train:
       images, labels = images.to(device), labels.to(device)
       output = model(images)
       loss = criterion(output, labels)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       epoc_loss += loss.item()
       
    print(f'Loss of Epoch {epoch+1} is {epoc_loss / len(loader_train)}')
