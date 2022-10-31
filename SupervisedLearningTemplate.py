import matplotlib.pyplot as plt
import numpy as np



class LinearRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def Cost(self, w, b):
        f_wb = 0
        X_train = self.X
        m = len(X_train)
        y = self.y

        cost = 0
        for i in range(m):
            f_wb = np.dot(w, X_train[i]) + b
            cost += (f_wb - y[i])**2

        return cost/(2*m)


    def Gradient(self, w, b):
        dj_dw = 0
        dj_db = 0

        X_train = self.X
        y = self.y
        m = len(X_train)

        for i in range(m):
            f_wb = np.dot(X_train[i],w) + b
            error = f_wb - y[i]
            dj_dw += error * X_train[i]
            dj_db += error

        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db

    
    def GradientDescent(self, alpha, nb_iters = 10000):
        w_in = 0
        b_in = 0

        for i in range(nb_iters):
            dj_dw, dj_db = self.Gradient(w_in, b_in)
            w_in = w_in - alpha * dj_dw
            b_in = b_in - alpha * dj_db
        

        return w_in, b_in


    def plot(self, w=0, b=0):
        plt.close('all')
        X_train = self.X
        plt.plot(X_train, self.y)
        predict = w*X_train+b
        plt.plot(X_train, predict)
        plt.show()



X = np.ndarray(shape=(3,1), buffer=np.array([1.00, 2.00, 3.00]))
y = np.ndarray(shape=(3,1), buffer=np.array([1.00, 2.50, 3.50]))

lr = LinearRegression(X, y)
w, b = lr.GradientDescent(0.01)
print(lr.plot(w, b))