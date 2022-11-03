import matplotlib.pyplot as plt
import numpy as np, csv
import statistics



class Regression:
    def __init__(self, csv_file):
        
        X, y = self.from_csv_to_array(csv_file)
        
        self.X_train = X
        self.y = y
        self.m, self.n = X.shape
    


    def from_csv_to_array(self, csv_file):
        with open(csv_file, "r") as f:
            data = csv.reader(f)
            data = list(data)

        data = data[1:400] # remove titles
        
        # convert string numbers to floats
        X = np.array([list(map(float, data[i][1:-1])) for i in range(len(data))]) 
        y = np.array([float(data[i][-1]) for i in range(len(data))])
                
        return X, y

    

    def feature_scaling(self):
        for i in range(self.m):
            mean = statistics.mean(self.X_train[i])
            std_der = statistics.stdev(self.X_train[i])
            self.X_train[i] = (self.X_train[i] - mean)/std_der




    def Cost(self, w, b):
        cost = 0
        for i in range(self.m):
            f_wb = np.dot(w, self.X_train[i]) + b # works for multiple features using vectorization
            cost += (f_wb - self.y[i])**2

        return cost/(2*self.m)


    def Gradient(self, w, b):
        dj_dw = np.zeros(self.n)
        dj_db = 0

        for i in range(self.m):

            error = np.dot(self.X_train[i], w) + b - self.y[i]

            for j in range(self.n):
                dj_dw[j] += error*self.X_train[i, j]
            dj_db += error

        dj_dw /= self.m
        dj_db /= self.m

        return dj_dw, dj_db

    
    def GradientDescent(self, alpha):
        w = np.zeros(self.n)
        b = 0
        epsilon = 1e-3
        error, prev_err = 0, 1

        while(1):
            error = self.Cost(w, b)
            if (abs(prev_err - error) < epsilon):
                return w, b
            prev_err = error
            dj_dw, dj_db = self.Gradient(w, b)
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
        

    def plot(self, w=0, b=0):
        plt.close('all')
        plt.plot(self.X_train, self.y)
        predict = w*self.X_train + b
        plt.plot(self.X_train, predict)
        plt.show()
    

    def predict(self, w, b, x_pred):
        return np.dot(w, x_pred) + b



lr = Regression("adm_data.csv")
#lr.feature_scaling()
w, b = lr.GradientDescent(0.01)

print(f"W: {w}\nb: {b}")

test = lr.X_train[0]
print(test)

print(lr.predict(w, b, test))