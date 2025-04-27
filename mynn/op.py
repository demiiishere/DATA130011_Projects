from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self):
        self.training = True
        self.optimizable = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

def he_init(size):
    fan_in = size[0]
    return np.random.randn(*size) * np.sqrt(2. / fan_in)

class Linear(Layer):
    def __init__(self, in_dim, out_dim, initialize_method=he_init, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))  # [in_dim, out_dim]
        self.b = initialize_method(size=(1, out_dim))       # [1, out_dim]
        self.grads = {'W': None, 'b': None}
        self.input = None  # Record input for backward

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.optimizable = True

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input = X  # Cache for backward
        output = np.dot(X, self.W) + self.b  # Broadcasting handles bias
        return output

    def backward(self, grad: np.ndarray):
        batch_size = self.input.shape[0]

        # Gradients w.r.t parameters
        grad_W = np.dot(self.input.T, grad) / batch_size  # [in_dim, out_dim]
        grad_b = np.sum(grad, axis=0, keepdims=True) / batch_size  # [1, out_dim]

        # Weight decay (L2 regularization)
        if self.weight_decay:
            grad_W += self.weight_decay_lambda * self.W

        self.grads['W'] = grad_W
        self.grads['b'] = grad_b

        # Gradients w.r.t input to pass back
        grad_input = np.dot(grad, self.W.T)  # [batch_size, in_dim]
        return grad_input

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=np.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.optimizable = True

        # initialize parameters
        self.W = initialize_method(size=(out_channels, in_channels, *self.kernel_size))  # [out, in, k, k]
        self.b = np.zeros((out_channels, 1)) 

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)


    def forward(self, X):

        if self.training:
            self.input = X

        B, C, H, W = X.shape
        K, _, kH, kW = self.W.shape

        out_H = (H + 2 * self.padding - kH) // self.stride + 1
        out_W = (W + 2 * self.padding - kW) // self.stride + 1

        # padding
        if self.padding > 0:
            X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # im2col:
        cols = []
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                w_start = j * self.stride
                patch = X[:, :, h_start:h_start + kH, w_start:w_start + kW]  # [B, C, kH, kW]
                cols.append(patch.reshape(B, -1))  # --> [B, C*kH*kW]
        X_col = np.stack(cols, axis=-1)  # [B, C*kH*kW, out_H * out_W]

        # reshape W to matrix: [K, C*kH*kW]
        W_col = self.W.reshape(K, -1)  # [K, C*kH*kW]
        out = np.einsum('kc,bcp->bkp', W_col, X_col) + self.b.reshape(1, K, 1)
        out = out.reshape(B, K, out_H, out_W)  # [B, K, H_out, W_out]
        if self.training:
            self.X_padded = X
            self.X_col = X_col
        return out
    


    def backward(self, grads):
        B, K, H_out, W_out = grads.shape
        kH, kW = self.kernel_size
        X_col = self.X_col  # [B, C*kH*kW, H_out * W_out]
        W_col = self.W.reshape(K, -1)  # [K, C*kH*kW]

        # reshape grads: [B, K, H_out, W_out] → [B, K, H_out * W_out]
        grads_reshaped = grads.reshape(B, K, -1)

        # dW: [K, C*kH*kW]
        dW = np.einsum('bkp,bcp->kc', grads_reshaped, X_col) / B
        dW = dW.reshape(self.W.shape)  
        db = np.sum(grads_reshaped, axis=(0, 2), keepdims=True).reshape(self.b.shape) / B

        # dX_col: [B, C*kH*kW, H_out * W_out]
        dX_col = np.einsum('kc,bkp->bcp', W_col, grads_reshaped)

        # initialize dX_padded
        B, C, H_padded, W_padded = self.X_padded.shape
        dX_padded = np.zeros((B, C, H_padded, W_padded))

        # †ranspose of im2col: dX_col-->dX_padded
        out_idx = 0
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                patch = dX_col[:, :, out_idx]  # shape [B, C*kH*kW]
                patch = patch.reshape(B, C, kH, kW)
                dX_padded[:, :, h_start:h_start + kH, w_start:w_start + kW] += patch
                out_idx += 1

        # delete padding
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        # weight decay
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        # save gradient
        self.grads['W'] = dW
        self.grads['b'] = db

        return dX
    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MultiCrossEntropyLoss(Layer):
    def __init__(self, model=None, max_classes=10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True  # with softmax by default
        self.preds = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.labels = labels
        if self.has_softmax:
            # do softmax
            exps = np.exp(predicts - np.max(predicts, axis=1, keepdims=True))
            self.preds = exps / np.sum(exps, axis=1, keepdims=True)
        else:
            self.preds = predicts  

        batch_size = predicts.shape[0]
        # add epsilon
        epsilon = 1e-12
        log_probs = -np.log(self.preds[np.arange(batch_size), labels] + epsilon)
        loss = np.mean(log_probs)
        return loss


    def backward(self):
        batch_size = self.preds.shape[0]
        self.grads = self.preds.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    

class MultiMSELoss(Layer):
    def __init__(self, model=None) -> None:
        self.model = model
        self.preds = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.preds = predicts
        self.labels = labels

        batch_size = predicts.shape[0]
        one_hot_labels = np.zeros_like(predicts)
        one_hot_labels[np.arange(batch_size), labels] = 1

        self.one_hot_labels = one_hot_labels  
        loss = np.mean((self.preds - one_hot_labels) ** 2)
        return loss

    def backward(self):
        batch_size = self.preds.shape[0]
        self.grads = (2 * (self.preds - self.one_hot_labels)) / batch_size

        self.model.backward(self.grads)


    
class L2Regularization(Layer):
    def __init__(self, model=None, weight_decay=1e-4) -> None:
        self.model = model
        self.weight_decay = weight_decay
        self.loss = 0.0
        self.grads = None

    def __call__(self):
        return self.forward()

    def forward(self):
        """
        Compute the L2 loss: sum of squared weights times weight_decay.
        """
        l2_sum = 0.0
        for layer in self.model.layers:  # 假设你的 model 有 layers 列表
            if hasattr(layer, 'params'):
                for param in layer.params.values():
                    l2_sum += np.sum(param ** 2)
        self.loss = 0.5 * self.weight_decay * l2_sum
        return self.loss

    def backward(self):
        """
        Add L2 gradient: derivative of (1/2 * weight_decay * ||w||^2) is weight_decay * w
        """
        for layer in self.model.layers:
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params.keys():
                    if layer.grads.get(key) is not None:
                        layer.grads[key] += self.weight_decay * layer.params[key]

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

import numpy as np

import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.optimizable = False
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if not self.training:
            return X
        self.mask = (np.random.rand(*X.shape) > self.p) / (1.0 - self.p)
        return X * self.mask

    def backward(self, grad_output):
        if not self.training:
            return grad_output
        return grad_output * self.mask

    def train(self):
        self.training = True

    def eval(self):
        self.training = False