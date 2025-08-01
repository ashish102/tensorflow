import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def int_to_binary_tensor(n: int, num_bits: int = 17) -> torch.Tensor:
    return torch.tensor([(n >> i) & 1 for i in range(num_bits)], dtype=torch.float32)


class PrimeDataset(Dataset):
    def __init__(self, start: int, end: int, num_bits: int = 17) -> None:
        self.data = [int_to_binary_tensor(n, num_bits) for n in range(start, end)]
        self.labels = torch.tensor([1 if is_prime(n) else 0 for n in range(start, end)], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


class PrimeNet(nn.Module):
    def __init__(self, num_bits: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_bits, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x).squeeze()
            pred = (output > 0.5).long()
            correct += (pred == y.long()).sum().item()
            total += y.size(0)
    return correct / total


def main() -> None:
    num_bits = 17
    train_ds = PrimeDataset(1, 10001, num_bits)
    test_ds = PrimeDataset(10000, 100001, num_bits)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrimeNet(num_bits).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}: train loss {avg_loss:.4f}")

    accuracy = evaluate(model, test_loader, device)
    print(f"Test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
