import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
from model import SimpleVGG6

def train_and_validate(config=None):
    """Run training and validation using W&B sweep config."""

    with wandb.init(config=config, project="vgg6_sweep_Ansuman"):
        cfg = wandb.config

        act_map = {
            "Sigmoid": nn.Sigmoid,
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "SiLU": nn.SiLU,
            "GELU": nn.GELU,
        }
        opt_map = {
            "Adam": lambda p, lr: torch.optim.Adam(p, lr=lr),
            "Adagrad": lambda p, lr: torch.optim.Adagrad(p, lr=lr),
            "SGD": lambda p, lr: torch.optim.SGD(p, lr=lr),
            "Nesterov": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True),
            "RMSprop": lambda p, lr: torch.optim.RMSprop(p, lr=lr),
            "Nadam": lambda p, lr: torch.optim.NAdam(p, lr=lr),
        }

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleVGG6(act_map[cfg.activation]).to(device)
        optimizer = opt_map[cfg.optimizer](model.parameters(), lr=cfg.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        for epoch in range(cfg.epochs):
            model.train()
            total_train_loss, total_correct, total_samples = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                _, predicted = preds.max(1)
                total_samples += labels.size(0)
                total_correct += predicted.eq(labels).sum().item()

            train_acc = 100.0 * total_correct / total_samples
            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            val_loss, val_correct, val_samples = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = outputs.max(1)
                    val_samples += labels.size(0)
                    val_correct += preds.eq(labels).sum().item()

            val_acc = 100.0 * val_correct / val_samples
            avg_val_loss = val_loss / len(test_loader)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "vgg6_best.pth")
                artifact = wandb.Artifact("vgg6_best_model", type="model")
                artifact.add_file("vgg6_best.pth")
                wandb.log_artifact(artifact)
