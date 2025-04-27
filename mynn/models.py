from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout=False):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout = dropout
        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])

                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]

                self.layers.append(layer)

                if i < len(size_list) - 2:
                    if act_func == 'Logistic':
                        raise NotImplementedError
                    elif act_func == 'ReLU':
                        layer_f = ReLU()
                    self.layers.append(layer_f)

                    if self.dropout:
                        self.layers.append(Dropout(p=0.2)) 

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Flatten(Layer):
    """
    Flatten layer: [B, C, H, W] -> [B, C * H * W]
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)
    

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.argmax_mask = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        B, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride

        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1

        shape = (B, C, out_H, out_W, k, k)
        strides = (
            X.strides[0],
            X.strides[1],
            X.strides[2] * s,
            X.strides[3] * s,
            X.strides[2],
            X.strides[3],
        )
        patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        # shape: [B, C, H_out, W_out, k, k]

        reshaped = patches.reshape(B, C, out_H, out_W, -1)
        out = reshaped.max(axis=-1)  # [B, C, H_out, W_out]
        if self.training:
            max_mask = reshaped == out[..., None]  # [B, C, H_out, W_out, k*k]
            self.argmax_mask = max_mask.reshape(patches.shape)

        return out
    def backward(self, grads):
        B, C, H, W = self.input.shape
        k = self.kernel_size
        s = self.stride
        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1

        dX = np.zeros_like(self.input)
        mask = self.argmax_mask.reshape(B, C, out_H, out_W, k * k)
        grads_expand = grads[..., None]  # [B, C, H_out, W_out, 1]

        grads_broadcasted = grads_expand * mask  # 只对最大位置反传梯度
        grads_broadcasted = grads_broadcasted.reshape(B, C, out_H, out_W, k, k)

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * s
                w_start = j * s
                dX[:, :, h_start:h_start+k, w_start:w_start+k] += grads_broadcasted[:, :, i, j, :, :]

        return dX
    

class Model_CNN(Layer):
    """
    A simple CNN model: Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Flatten → Linear → ReLU → Linear
    """
    def __init__(self):
        super().__init__()
        self.layers = []

        # self.layers.append(conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0))
        # self.layers.append(ReLU())
        # self.layers.append(MaxPool2D(kernel_size=2, stride=2))
        # self.layers.append(conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0))
        # self.layers.append(ReLU())
        # self.layers.append(MaxPool2D(kernel_size=2, stride=2))

        # self.layers.append(Flatten())
        # self.layers.append(Linear(in_dim=16 * 5 * 5, out_dim=100))
        # self.layers.append(ReLU())
        # self.layers.append(Linear(in_dim=100, out_dim=10)) 

        # if we do padding:
        self.layers.append(conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(kernel_size=2, stride=2))
        self.layers.append(conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(kernel_size=2, stride=2))

        self.layers.append(Flatten())
        self.layers.append(Linear(in_dim=16 * 7 * 7, out_dim=100))
        self.layers.append(ReLU())
        self.layers.append(Linear(in_dim=100, out_dim=10)) 
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def train(self):
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.eval()
    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        cnt = 0
        for layer in self.layers:
            if hasattr(layer, 'optimizable') and layer.optimizable:
                layer.W = param_list[cnt]['W']
                layer.b = param_list[cnt]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[cnt]['weight_decay']
                layer.weight_decay_lambda = param_list[cnt]['lambda']
                cnt += 1

    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if hasattr(layer, 'optimizable') and layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)