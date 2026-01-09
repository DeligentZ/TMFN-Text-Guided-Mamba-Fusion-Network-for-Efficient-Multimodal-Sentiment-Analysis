import yaml
import random
from trainer import Trainer
from data_loader import data_loader
from models.multimodal import MultimodalModel
import torch
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"\ndevice is available: {device}\n")

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure training repeatability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    config = load_config("configs/default.yaml")
    set_seed(config["seed"])
    train_loader, val_loader, test_loader = data_loader(config)
    model = MultimodalModel(config).to(device)
    trainer = Trainer(config)
    results = (trainer.fit(model, train_loader, val_loader, test_loader))

if __name__ == "__main__":
    main()