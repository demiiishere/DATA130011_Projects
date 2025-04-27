# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import scipy.ndimage

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]


def augment_images(images, labels, num_augments=1):
    augmented_images = []
    augmented_labels = []
    for i in range(len(images)):
        img = images[i].reshape(28, 28)
        for _ in range(num_augments):
            angle = np.random.uniform(-15, 15)
            rotated = scipy.ndimage.rotate(img, angle, reshape=False, mode='nearest')

            shift_x = np.random.uniform(-2, 2)
            shift_y = np.random.uniform(-2, 2)
            shifted = scipy.ndimage.shift(rotated, shift=(shift_x, shift_y), mode='nearest')

            zoom = np.random.uniform(0.9, 1.1)
            zoomed = scipy.ndimage.zoom(shifted, zoom)
            if zoomed.shape[0] > 28:
                start = (zoomed.shape[0] - 28) // 2
                zoomed = zoomed[start:start+28, start:start+28]
            elif zoomed.shape[0] < 28:
                pad = (28 - zoomed.shape[0]) // 2
                zoomed = np.pad(zoomed, ((pad, 28 - zoomed.shape[0] - pad), (pad, 28 - zoomed.shape[1] - pad)), mode='constant')

            augmented_images.append(zoomed.flatten())
            augmented_labels.append(labels[i])
    
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    return augmented_images, augmented_labels

# # data argumentation
# aug_imgs, aug_labs = augment_images(train_imgs, train_labs, num_augments=1)

# train_imgs = np.vstack([train_imgs, aug_imgs])
# train_labs = np.hstack([train_labs, aug_labs])

# print(f"Training data expanded to {train_imgs.shape[0]} samples.")

# MLP Model version:
#  normalize from [0, 255] to [0, 1]
train_imgs = train_imgs.astype(np.float32) / 255.
valid_imgs = valid_imgs.astype(np.float32) / 255.

# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', None, dropout=True)

optimizer = nn.optimizer.SGD(init_lr=1.2, model=linear_model)
# optimizer = nn.optimizer.MomentGD(init_lr=1.2, model=linear_model, mu=0.99)

# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=300)
# scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer)

# loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
loss_fn = nn.op.MultiMSELoss(model=linear_model)
# l2_reg = nn.op.L2Regularization(model=linear_model, weight_decay=1e-4)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, l2_reg=None)
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=10, log_iters=100, save_dir=r'./best_models')


# # CNN model version:
# train_imgs = train_imgs.reshape(-1, 1, 28, 28).astype(np.float32) / 255.
# valid_imgs = valid_imgs.reshape(-1, 1, 28, 28).astype(np.float32) / 255.
# cnn_model = nn.models.Model_CNN()

# optimizer = nn.optimizer.MomentGD(init_lr=1.2, model=cnn_model, mu=0.9)
# # optimizer = nn.optimizer.SGD(init_lr=0.12, model=cnn_model) # (0.12--batch size 128)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
# loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

# runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')


_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()