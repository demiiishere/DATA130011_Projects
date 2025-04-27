from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]
                
                if 'W' in layer.params:
                    layer.W = layer.params['W']
                if 'b' in layer.params:
                    layer.b = layer.params['b']

class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        # Initialize velocity only for layers with parameters
        self.velocity = {layer: {key: np.zeros_like(value) for key, value in layer.params.items()} 
                         for layer in self.model.layers if hasattr(layer, 'params')}

    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, 'params') and layer.optimizable:  # Check if the layer has parameters
                for key in layer.params.keys():
                    # Update velocity with momentum
                    self.velocity[layer][key] = self.mu * self.velocity[layer][key] + (1 - self.mu) * layer.grads[key]
                    
                    # Apply the momentum gradient update
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    
                    layer.params[key] -= self.init_lr * self.velocity[layer][key]

                if 'W' in layer.params:
                    layer.W = layer.params['W']
                if 'b' in layer.params:
                    layer.b = layer.params['b']