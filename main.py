import wandb
from train_validate import train_and_validate

wandb.login()

sweep_cfg = {
    "method": "random",
    "metric": {"name": "Validation Accuracy", "goal": "maximize"},
    "parameters": {
        "activation": {"values": ["ReLU", "Sigmoid", "Tanh", "SiLU", "GELU"]},
        "optimizer": {"values": ["SGD", "Nesterov", "Adam", "Adagrad", "RMSprop", "Nadam"]},
        "batch_size": {"values": [64, 128, 256, 512]},
        "epochs": {"values": [10, 30, 60, 80, 100]},
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
    },
}

sweep_id = wandb.sweep(sweep_cfg, project="vgg6_sweep_Ansuman")
wandb.agent(sweep_id, function=train_and_validate)
