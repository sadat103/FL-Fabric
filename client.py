# client.py
import os
import json
import time
import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, ResNet50_Weights,ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import flwr as fl
from flwr.client import start_client


# ─── Data loading ────────────────────────────────────────────────────────────────

class JSONImageDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, labels_file)) as f:
            raw = json.load(f)

        # Build filename→label mapping
        if isinstance(raw, dict):
            self.labels = raw
        elif isinstance(raw, list) and isinstance(raw[0], dict):
            self.labels = {entry["filename"]: entry["label"] for entry in raw}
        elif isinstance(raw, list):
            fnames = sorted(
                fn for fn in os.listdir(data_dir)
                if fn.lower().endswith((".jpg", ".png", ".jpeg"))
            )
            if len(raw) != len(fnames):
                raise ValueError(f"Got {len(raw)} labels but {len(fnames)} images")
            self.labels = dict(zip(fnames, raw))
        else:
            raise ValueError(f"Unrecognized labels.json format: {type(raw)}")

        classes = sorted(set(self.labels.values()))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.filenames = list(self.labels.keys())
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        cls = self.labels[fname]
        img = Image.open(os.path.join(self.data_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[cls]


def make_data_loaders(data_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ])
    train_ds = JSONImageDataset(
        data_dir=os.path.join(data_dir, "train"),
        labels_file="labels.json",
        transform=transform,
    )
    test_ds = JSONImageDataset(
        data_dir=os.path.join(data_dir, "test"),
        labels_file="labels.json",
        transform=transform,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Data: {len(train_ds)} train, {len(test_ds)} test samples")
    return train_loader, test_loader


# ─── Model definition ────────────────────────────────────────────────────────────

def get_resnet50_custom(num_classes: int) -> nn.Module:
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for p in base.parameters():
        p.requires_grad = False
    layers = list(base.children())[:-2]
    backbone = nn.Sequential(*layers)
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(base.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return nn.Sequential(backbone, head)


def get_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "SimpleCNN":
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2,2)
                self.fc1 = nn.Linear(32*64*64, num_classes)
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = x.view(-1, 32*64*64)
                return self.fc1(x)
        return SimpleCNN()

    elif model_name == "ResNet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        return model

    elif model_name == "ResNet50":
        return get_resnet50_custom(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ─── Training & Evaluation ───────────────────────────────────────────────────────

def train(model: nn.Module, loader: DataLoader, epochs: int) -> dict:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        steps_per_epoch=len(loader), epochs=epochs
    )
    for _ in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model.state_dict()


def evaluate_model(model: nn.Module, parameters, loader: DataLoader) -> float:
    # Load new params
    state = model.state_dict()
    for k, new_p in zip(state.keys(), parameters):
        state[k].data.copy_(torch.from_numpy(new_p))
    # Compute accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def compute_loss(model: nn.Module, loader: DataLoader) -> float:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            loss = criterion(model(images), labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    return total_loss / total_samples if total_samples else 0.0


# ─── FL Client with CSV logging ─────────────────────────────────────────────────

class TrackingClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, csv_path):
        self.model = model
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.csv_path = csv_path

        # Internal round counter
        self._round = 0
        self._last_train_time = 0.0

    def get_parameters(self, config):
        return [p.cpu().numpy() for _, p in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Bump round
        self._round += 1

        # Load global params
        sd = self.model.state_dict()
        for k, new_p in zip(sd.keys(), parameters):
            sd[k].copy_(torch.from_numpy(new_p))

        # Train
        start = time.time()
        new_state = train(self.model, self.train_loader, epochs=10)
        self._last_train_time = time.time() - start

        print(f"Round {self._round}: train time {self._last_train_time:.2f}s")
        return list(new_state.values()), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Load params
        sd = self.model.state_dict()
        for k, new_p in zip(sd.keys(), parameters):
            sd[k].copy_(torch.from_numpy(new_p))

        # Compute metrics
        loss = compute_loss(self.model, self.test_loader)
        acc = evaluate_model(self.model, parameters, self.test_loader)

        print(f"Round {self._round}: loss {loss:.4f}, acc {acc:.4f}")

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.model,
                self._round,
                acc,
                loss,
                self._last_train_time,
            ])

        return loss, len(self.test_loader.dataset), {"accuracy": acc}


# ─── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower FL Client")
    parser.add_argument("--dataset-dir", "-d", required=True,
                        help="Path to folder containing `train/labels.json` and `test/labels.json`")
    parser.add_argument("--model-name", "-m", default="ResNet18",
                        choices=["SimpleCNN", "ResNet18", "ResNet50"])
    parser.add_argument("--class-count", "-c", type=int, required=True,
                        help="Number of target classes")
    parser.add_argument("--server-ip", "-i", default="127.0.0.1",
                        help="FL server IP address")
    parser.add_argument("--server-port", "-p", default="8080",
                        help="FL server port")
    parser.add_argument("--type", "-t", default="IID",
                        help="Data Distribution")
    args = parser.parse_args()
    
    path = f"./results/{args.type}"
    
    if not os.path.exists(path):
            with open(f"{path}/{args.dataset_dir}_2024_evaluation.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["model","round", "accuracy", "loss", "training_time"])
    csv_path = f"{path}/evaluation.csv"
    
    # Prepare data & model
    train_loader, test_loader = make_data_loaders(f"./fl_client/{args.dataset_dir}")
    model = get_model(args.model_name, args.class_count)

    # Start Flower client
    client = TrackingClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        csv_path=csv_path,
    )

    try:
        start_client(
            server_address=f"{args.server_ip}:{args.server_port}",
            client=client.to_client(),
        )
    except Exception as e:
        start_client(
            server_address=f"{args.server_ip}:{args.server_port}",
            client=client.to_client(),
          )  

