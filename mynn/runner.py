import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=512, scheduler=None, l2_reg=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            self.model.train()

            X, y = train_set
            assert X.shape[0] == y.shape[0]
            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            num_batches = X.shape[0] // self.batch_size

            with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for iteration in range(num_batches):
                    train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                    train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                    logits = self.model(train_X)
                    trn_loss = self.loss_fn(logits, train_y)
                    if self.l2_reg is not None:
                        trn_loss += self.l2_reg.forward()
                    self.train_loss.append(trn_loss)
                    
                    trn_score = self.metric(logits, train_y)
                    self.train_scores.append(trn_score)

                    self.loss_fn.backward()
                    if self.l2_reg is not None:
                        self.l2_reg.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)

                    if (iteration) % log_iters == 0:
                        pbar.write(f"[Train] loss: {trn_loss:.4f}, score: {trn_score:.4f} | [Dev] loss: {dev_loss:.4f}, score: {dev_score:.4f}")

                    pbar.set_postfix({"train_loss": f"{trn_loss:.4f}", "train_score": f"{trn_score:.4f}"})
                    pbar.update(1)

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"Best accuracy updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score

        self.best_score = best_score


    def evaluate(self, data_set):
        self.model.eval()
        X, y = data_set
        eval_batch_size = 2048
        assert X.shape[0] == y.shape[0]
        num_batches = X.shape[0] // eval_batch_size

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i in range(num_batches):
            batch_X = X[i * eval_batch_size : (i + 1) * eval_batch_size]
            batch_y = y[i * eval_batch_size : (i + 1) * eval_batch_size]

            logits = self.model(batch_X)
            loss = self.loss_fn(logits, batch_y)

            preds = np.argmax(logits, axis=1)
            correct = np.sum(preds == batch_y)

            total_loss += loss * batch_X.shape[0]
            total_correct += correct
            total_samples += batch_X.shape[0]

        final_loss = total_loss / total_samples
        final_score = total_correct / total_samples  # accuracy
        return final_score, final_loss
    

    def save_model(self, save_path):
        self.model.save_model(save_path)