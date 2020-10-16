import numpy as np
import matplotlib.pyplot as plt

class Linear_Regression:

    def __init__(self):
        return

    def fit(self, X, Y):
        X_ = sum(X) / len(X)
        Y_ = sum(Y) / len(Y)
        Xi_Yi = sum([(x - X_) * (y - Y_) for x, y in zip(X, Y)])
        Xi_2 = sum([(x - X_) ** 2 for x in X])

        self.beta2 = Xi_Yi / Xi_2
        self.beta1 = Y_ - self.beta2 * X_

        return self.beta2, self.beta1

    def coord(self, x):
        return self.beta2 * x + self.beta1

    def predict(self,X):
        return [self.coord(x) for x in X]

