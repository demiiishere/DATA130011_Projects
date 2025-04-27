# you may do your own hyperparameter search job here.
import itertools
import pickle
import mynn as nn
from draw_tools.plot import plot

import numpy as np
import matplotlib.pyplot as plt
import gzip
from struct import unpack
import os

def load_mnist(train_images_path, train_labels_path):
    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    idx = np.random.permutation(np.arange(num))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    train_imgs = train_imgs.reshape(-1, 1, 28, 28).astype(np.float32) / 255.
    valid_imgs = valid_imgs.reshape(-1, 1, 28, 28).astype(np.float32) / 255.
    return train_imgs, train_labs, valid_imgs, valid_labs


def train_and_evaluate(hparams, save_dir):
    train_imgs, train_labs, valid_imgs, valid_labs = load_mnist(
        './dataset/MNIST/train-images-idx3-ubyte.gz',
        './dataset/MNIST/train-labels-idx1-ubyte.gz'
    )

    model = nn.models.Model_CNN()

    # Optimizer
    if hparams['optimizer'] == 'SGD':
        optimizer = nn.optimizer.SGD(init_lr=hparams['init_lr'], model=model)
    elif hparams['optimizer'] == 'Momentum':
        optimizer = nn.optimizer.MomentGD(init_lr=hparams['init_lr'], model=model, mu=hparams['mu'])

    # Scheduler
    if hparams['scheduler'] == 'StepLR':
        scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=hparams['step_size'])
    elif hparams['scheduler'] == 'MultiStepLR':
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=hparams['milestones'], gamma=hparams['gamma'])

    # Loss
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max() + 1)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=hparams['epochs'], log_iters=200, save_dir=save_dir)

    # return final validation's accuracy
    return runner.dev_scores[-1]


param_grid = {
    'init_lr': [0.1, 0.5, 1.2],
    'optimizer': ['SGD', 'Momentum'],
    'mu': [0.9],  
    'scheduler': ['StepLR', 'MultiStepLR'],
    'step_size': [300], 
    'milestones': [[800, 2400, 4000]], 
    'gamma': [0.5],
    'epochs': [5]
}

# generate all possible groups
keys = list(param_grid.keys())
values = list(param_grid.values())
all_combinations = list(itertools.product(*values))


results = []

for combo in all_combinations:
    hparams = dict(zip(keys, combo))

    if hparams['optimizer'] == 'SGD':
        hparams['mu'] = None
    if hparams['scheduler'] == 'StepLR':
        hparams['milestones'] = None
    if hparams['scheduler'] == 'MultiStepLR':
        hparams['step_size'] = None

    print(f"Training with hyperparameters: {hparams}")

    save_dir = './tuning_models'
    os.makedirs(save_dir, exist_ok=True)
    acc = train_and_evaluate(hparams, save_dir)
    print(f"Validation accuracy: {acc:.4f}")

    results.append((hparams, acc))


best_hparams, best_acc = max(results, key=lambda x: x[1])
print("\nBest Hyperparameters:")
for k, v in best_hparams.items():
    print(f"{k}: {v}")
print(f"Best Validation Accuracy: {best_acc:.4f}")