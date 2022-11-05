import matplotlib.pyplot as plt
import numpy as np, csv
import statistics
from scipy import stats



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'




class LinearRegression:
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
        # Manual Feature Scaling
        mean = statistics.mean(self.X_train[0])
        std_der = statistics.stdev(self.X_train[0])
        for i in range(self.m):
            self.X_train[i] = (self.X_train[i] - mean)/std_der

        # for i in range(self.m):
        #     self.X_train[i] = stats.zscore(self.X_train[i])




    def Cost(self, w, b, lambda_=1):
        cost = 0
        for i in range(self.m):
            f_wb = np.dot(w, self.X_train[i]) + b # works for multiple features using vectorization
            cost += (f_wb - self.y[i])**2
        
        cost /= (2*self.m)

        regular_term = 0
        for j in range(self.n):
            regular_term += w[j]**2
                
        regular_term *= (lambda_/(2*self.m))

        total_cost = cost + regular_term
        
        return total_cost


    def Gradient(self, w, b, lambda_=1):
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

    
    def GradientDescent(self, alpha=0.01, lambda_=1, epsilon = 1e-8):
        print("\n" + bcolors.BOLD + bcolors.FAIL + "[+] Training ...\n" + bcolors.ENDC)
        w = np.zeros(self.n)
        b = 0
        prev_err = self.Cost(w, b, lambda_)+1
        i=0

        # until the cost is decreasing really too slowly
        while(1):
            i+=1
            error = self.Cost(w, b, lambda_)
            if (abs(prev_err - error) < epsilon):
                print(f"{bcolors.BOLD + bcolors.OKGREEN}\n[-] W: {w}\n\n[-] b: {b}\n")
                return w, b
            prev_err = error
            if (i%1000 == 0):
                print(bcolors.WARNING + "[°] Squared Error Function: " + str(error) + "\n")
            dj_dw, dj_db = self.Gradient(w, b, lambda_)
            w = w*(1-((alpha*lambda_)/self.m)) - alpha * dj_dw # using the lambda factor for regularization
            b = b - alpha * dj_db
        

    def plot(self, w=0, b=0):
        plt.close('all')
        plt.plot(self.X_train, self.y)
        predict = w*self.X_train + b
        plt.plot(self.X_train, predict)
        plt.show()
    

    def predict(self, w, b, x_pred):
        return np.dot(w, x_pred) + b
    

    def accuracy(self, w, b, test):
        tot = 0
        epsilon = 1e-1
        
        for i in range(self.m):
            if (abs(self.predict(w, b, test[i]) - self.y[i]) < epsilon): tot+=1
        
        print(bcolors.BOLD + bcolors.UNDERLINE + bcolors.OKCYAN + "[+] Prediction Accuracy: " + str((tot/self.m)*100) + "%\n")
        return (tot/self.m)*100



lr = LinearRegression("adm_data.csv")

lr.feature_scaling()
w, b = lr.GradientDescent(0.01, 0.1)

acc = lr.accuracy(w, b, lr.X_train)