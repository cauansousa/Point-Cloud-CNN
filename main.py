import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Definindo o modelo de rede neural (simplificado PointNet)
class SimplePointNet(nn.Module):
    def __init__(self):
        super(SimplePointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)  # 3 características de entrada (x, y, z)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Saída binária (0 ou 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]  # Max pooling
        x = x.view(-1, 256)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid para classificação binária
        return x


# Criando um dataset de exemplo (10 nuvens de pontos, cada uma com 1000 pontos e 3 coordenadas)
num_samples = 10
num_points = 1000
point_clouds = torch.randn(num_samples, num_points, 3)  # Dados simulados
labels = torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)  # Rótulos binários (0 ou 1)

# DataLoader
dataset = TensorDataset(point_clouds, labels)
dataloader = DataLoader(dataset, batch_size=2)

# Inicializando o modelo, loss e otimizador
model = SimplePointNet()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
num_epochs = 50
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.transpose(1, 2)  # Transpor para corresponder à entrada da Conv1D
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Você pode adicionar validação, salvar o modelo, etc.
