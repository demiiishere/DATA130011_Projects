import pickle
import matplotlib.pyplot as plt

save_path = './best_models/best_model.pickle'
with open(save_path, 'rb') as f:
    param_list = pickle.load(f)

# --- first Conv2D ---
conv1_weight = param_list[0]['W']  # shape: (8, 1, 3, 3)

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
axes = axes.flatten()

for i in range(conv1_weight.shape[0]):
    kernel = conv1_weight[i, 0, :, :]
    ax = axes[i]
    im = ax.imshow(kernel, cmap='viridis')
    ax.set_title(f"Conv1 Kernel {i}", fontsize=8)
    ax.axis('off')

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()


# --- second Conv2D  ---
conv2_weight = param_list[1]['W']  # shape: (16, 8, 3, 3)

fig, axes = plt.subplots(4, 4, figsize=(10, 8))
axes = axes.flatten()

for i in range(conv2_weight.shape[0]):
    # 这里因为 in_channels=8，所以可以取平均一下
    kernel = conv2_weight[i].mean(axis=0)  # shape: (3,3)
    ax = axes[i]
    im = ax.imshow(kernel, cmap='viridis')
    ax.set_title(f"Conv2 Kernel {i}", fontsize=8)
    ax.axis('off')

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()



# --- second Linear ---
linear2_weight = param_list[3]['W']  # shape: (100, 10)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.flatten()

for i in range(10):  # 10 output classes
    neuron_weight = linear2_weight[:, i]  # shape: (100,)
    ax = axes[i]
    im = ax.imshow(neuron_weight.reshape(10, 10), cmap='viridis', aspect='auto')
    ax.set_title(f"Class {i}", fontsize=8)
    ax.axis('off')

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.tight_layout()
plt.subplots_adjust(right=0.9)
plt.show()