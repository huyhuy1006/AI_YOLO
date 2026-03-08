# train.py
import yaml
from training.trainer import Trainer
from utils.seed import set_seed

def main():
    set_seed(42)  # Cố định seed
    with open('configs/model.yaml', 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    with open('configs/data.yaml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    with open('configs/train.yaml', 'r', encoding='utf-8') as f:
        train_config = yaml.safe_load(f)

    trainer = Trainer(model_config, data_config, train_config)
    trainer.train(train_config['epochs'])

if __name__ == '__main__':
    main()