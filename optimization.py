import numpy as np
from oracles import BinaryHinge
import time


class PEGASOSMethod:
    def __init__(self, step_lambda, batch_size, num_iter):

        self.step_lambda = step_lambda
        self.batch_size = batch_size
        self.num_iter = num_iter
        
    def fit(self, X, y, trace=False):
        if trace is True:
            history = {}

        self.w = np.zeros(X.shape[1])
        w_best = np.copy(self.w)
        F_best = self.get_objective(X, y)
        if trace is True:
            history['func'] = [F_best]
            history['time'] = [0]
            start = time.clock()
        iteration = 1    
        
        all_indexes = np.random.choice(X.shape[0], self.batch_size * self.num_iter)
        while iteration <= self.num_iter:
            indexes = all_indexes[(iteration-1) * self.batch_size:iteration * self.batch_size]
            alpha = 1 / (iteration * self.step_lambda)
            self.w = ((1 - alpha * self.step_lambda) * self.w - 
                      alpha * self.get_gradient(X[indexes], y[indexes]))
            self.w = min(1, 1 / (self.step_lambda ** 0.5 * np.linalg.norm(self.w))) * self.w
            f = self.get_objective(X, y)
            if f < F_best:
                F_best = f
                w_best = np.copy(self.w)
            if trace is True:
                    history['func'].append(f)
                    history['time'].append(time.clock() - start)
                    start = time.clock()
            iteration += 1
        self.w = w_best
        if trace is True:
            return history 
        else:
            return self
        
    def get_objective(self, X, y):

        M = 1 - y * X.dot(self.w)
        M[M < 0] = 0
        return (1 / X.shape[0]) * M.sum() + 0.5 * self.step_lambda * self.w.dot(self.w.T)
    
    def get_gradient(self, X, y):
        
        M = 1 - y * X.dot(self.w)
        return (- 1 / X.shape[0]) * X.T.dot(y * np.where(M > 0, 1, 0)) + self.step_lambda * self.w
    
    def predict(self, X):
        return np.where(X.dot(self.w) > 0, 1, -1)

    
class GDClassifier:
    def __init__(self, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, C=1):        
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.C = C
        
    def get_params(self, deep=False):
        return {'C': self.C, 'step_alpha': self.step_alpha, 
                'step_beta': self.step_beta, 'tolerance': self.tolerance, 'max_iter': self.max_iter}
        
    def fit(self, X, y, w_0=None, trace=False):
        if trace is True:
            history = {}

        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0

        self.oracle = BinaryHinge(self.C)
        
        f = self.get_objective(X, y)
        F_best = f
        w_best = np.copy(self.w)
        if trace is True:
            history['func'] = [f]
            history['time'] = [0]
            start = time.clock()
        iteration = 1    
        f_pred = f
        self.w = self.w - self.step_alpha / (iteration ** self.step_beta) * self.get_gradient(X, y)
        f = self.get_objective(X, y)
        if f < F_best:
            F_best = f
            w_best = np.copy(self.w)
        if trace is True:
            history['func'].append(f)
            history['time'].append(time.clock() - start)
            start = time.clock()
        iteration += 1
        
        while (iteration <= self.max_iter and abs(f - f_pred) > self.tolerance):
            f_pred = f
            self.w = (self.w - self.step_alpha / (iteration ** self.step_beta) *
                      self.get_gradient(X, y))
            f = self.get_objective(X, y)
            if f < F_best:
                F_best = f
                w_best = np.copy(self.w)
            if trace is True:
                history['func'].append(f)
                history['time'].append(time.clock() - start)
                start = time.clock()
            iteration += 1
        self.w = w_best
        if trace is True:
            return history 
        else:
            return self
                    
    def predict(self, X):
        return np.where(X.dot(self.w) > 0, 1, -1)
            
    def get_objective(self, X, y):
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        return self.oracle.grad(X, y, self.w)
    
    def get_weights(self): 
        return self.w
        
        
class SGDClassifier(GDClassifier):
    
    def __init__(self, batch_size=1, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, random_seed=153, C=1):

        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.C = C
        
    def get_params(self, deep=False):
        return {'C': self.C, 'batch_size': self.batch_size, 'step_alpha': self.step_alpha, 
                'step_beta': self.step_beta, 'tolerance': self.tolerance, 'max_iter': self.max_iter,
                'random_seed': self.random_seed}  
    
    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        
        np.random.seed(self.random_seed)
        if trace is True:
            history = {}

        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0

        self.oracle = BinaryHinge(self.C)
        
        f = self.get_objective(X, y)
        F_best = f
        w_best = np.copy(self.w)
        epoch_num = 0
        if trace is True:
            history['epoch_num'] = [epoch_num]
            history['func'] = [f]
            history['time'] = [0]
            start = time.clock()
        iteration = 1    
        epoch_num_pred = epoch_num
        
        all_indexes = np.random.choice(X.shape[0], self.batch_size * self.max_iter)
        while iteration <= self.max_iter:
            indexes = all_indexes[(iteration-1) * self.batch_size:iteration * self.batch_size]
            self.w = (self.w - self.step_alpha / (iteration ** self.step_beta) * 
                      self.get_gradient(X[indexes], y[indexes]))
            epoch_num = self.batch_size * iteration / X.shape[0]
            if (epoch_num - epoch_num_pred) >= log_freq:
                f_pred = f
                epoch_num_pred = epoch_num
                f = self.get_objective(X, y)
                if f < F_best:
                    F_best = f
                    w_best = np.copy(self.w)
                if trace is True:
                    history['epoch_num'].append(epoch_num)
                    history['func'].append(f)
                    history['time'].append(time.clock() - start)
                    start = time.clock()
                if abs(f - f_pred) < self.tolerance:
                    print('Out!')
                    break
            iteration += 1
        self.w = w_best
        if trace is True:
            return history 
        else:
            return self  
