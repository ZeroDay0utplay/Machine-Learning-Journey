import matplotlib.pyplot as plt
import numpy as np, csv
import statistics
from sklearn import datasets
from scipy import stats
from time import sleep
import pandas as pd
from sklearn.model_selection import train_test_split



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
    PURPLE = "\033[95m"

class bg:
    black = '\033[40m'
    red = '\033[41m'
    green = '\033[42m'
    orange = '\033[43m'
    blue = '\033[44m'
    purple = '\033[45m'
    cyan = '\033[46m'
    lightgrey = '\033[47m'


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

        data = data[1:len(data)] # remove titles
        
        # convert string numbers to floats
        X = np.array([list(map(float, data[i][:-1])) for i in range(len(data))]) 
        y = np.array([float(data[i][-1]) for i in range(len(data))])
                
        return X, y

    

    def feature_scaling(self):
        # # Manual Feature Scaling
        # mean = statistics.mean(self.X_train)
        # std_der = statistics.stdev(self.X_train)
        # for i in range(self.m):
        #     self.X_train[i] = (self.X_train[i] - mean)/std_der

        mean_X = (sum(self.X_train))/self.m
        std_dev_X = np.sqrt(sum((self.X_train-mean_X)**2)/self.m)
        
        mean_y = (sum(self.y))/self.m
        std_dev_y = np.sqrt(sum((self.y-mean_y)**2)/self.m)

        for i in range(self.m):
            self.X_train[i][0] = (self.X_train[i][0] - mean_X)/std_dev_X
            self.y[i] = (self.y[i] - mean_y)/std_dev_y
        



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

    
    def Train(self, alpha=0.01, lambda_=1, epsilon = 1e-8):
        print("\n" + bcolors.BOLD + bcolors.PURPLE + "[+] Training ...\n" + bcolors.ENDC)
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
            dj_dw, dj_db = self.Gradient(w, b)
            w = w*(1-((alpha*lambda_)/self.m)) - alpha * dj_dw # using the lambda factor for regularization
            b = b - alpha * dj_db
        

    def plot(self, w=0, b=0):
        plt.close('all')
        plt.plot(self.X_train, self.y)
        plt.plot(self.X_train, self.predict(w, b, self.X_train))
        plt.show()
    

    def predict(self, w, b, x_pred):
        return np.dot(w, x_pred) + b
    

    def accuracy(self, w, b, test):
        tot = 0
        epsilon = 1e-1
        
        for i in range(self.m):
            if (abs(self.predict(w, b, test[i]) - self.y[i]) < epsilon): tot+=1
        
        print(bcolors.BOLD + bcolors.UNDERLINE + bcolors.OKCYAN + "[+] Prediction Accuracy: " + str((tot/self.m)*100) + "%\n" + bcolors.ENDC)
        return (tot/self.m)*100

















class LogisticRegression:
    def __init__(self, csv_file):
        
        self.from_csv_to_array(csv_file)
        self.update_vars()
        
    
    def update_vars(self):
        n = self.df.shape[1]-1
        self.X = np.array(self.df.drop(self.df.columns[[n]], axis=1))
        self.y = np.array(self.df.drop(self.df.columns[[i for i in range(n)]], axis=1))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.m, self.n = self.X_train.shape


    def from_csv_to_array(self, csv_file):
        df = pd.read_csv(csv_file)
        df = df.dropna()
        self.df = df
    


    def drop_label(self, labels):
        for label in labels:
            self.df = self.df[self.df['class'] != label]
        
        self.update_vars()
    

    
    def change_label_names(self, old_labels, new_labels):
        self.df['class'].replace(old_labels, new_labels, inplace=True)

        self.update_vars()
    

    def label_encoding(self, labels):
        self.df['class'].replace(labels, [1,0], inplace=True)
        
        self.update_vars()


    

    def feature_scaling(self):
        # Manual Feature Scaling
        self.X_train = stats.zscore(self.X_train)
    

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    

    
    def Cost(self, w, b, lambda_=1):
        cost = 0
        for j in range(self.m):
            f_wb = self.sigmoid(np.dot(self.X_train[j], w)+b)
            cost += (self.y_train[j]*np.log(f_wb) + (1-self.y_train[j])*np.log(1-f_wb))
        
        cost = -1/(self.m)*cost
        regular_term = 0
        for j in range(self.n):
            regular_term += w[j]**2
                
        regular_term *= (lambda_/(2*self.m))

        total_cost = cost + regular_term

        return total_cost
    

    def G(self, w, b):
        dj_dw = np.zeros((1,self.n))
        dj_db = 0
        for i in range(self.m):
            f_wb = self.sigmoid(np.dot(self.X_train[i], w)+b)
            error = self.y_train[i]*np.log(f_wb) + (1-self.y_train[i])*np.log(1-f_wb)
            for j in range(self.n):
                dj_dw[j] += error*self.X_train[i, j]
            dj_db += error
        
        dj_dw /= self.m
        dj_db /= self.m

        return dj_dw, dj_db

    

    def Train(self, learning_rate=0.01, no_iterations=30000):
        w = np.zeros((1, self.n))
        b = 0
        costs = []
        for i in range(no_iterations):
            #
            grads, cost = self.Gradient(w,b)
            #
            dw = grads["dw"]
            db = grads["db"]
            #weight update
            w = w - (learning_rate * (dw.T))
            b = b - (learning_rate * db)
            #
            
            if (i % 100 == 0):
                costs.append(cost)
                #print("Cost after %i iteration is %f" %(i, cost))
        
        #final parameters
        coeff = {"w": w, "b": b}
        gradient = {"dw": dw, "db": db}
        
        return coeff, gradient, costs


    def Gradient(self, w, b):
        m = self.X_train.shape[0]
        
        #Prediction
        final_result = self.sigmoid(np.dot(w,self.X_train.T)+b)
        Y_T = self.y_train.T
        cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
        #
        
        #Gradient calculation
        dw = (1/m)*(np.dot(self.X_train.T, (final_result-Y_T).T))
        db = (1/m)*(np.sum(final_result-Y_T))
        
        grads = {"dw": dw, "db": db}
        
        return grads, cost



    def plot(self, w=0, b=0):
        plt.close('all')
        plt.plot(self.X_train, self.y_train)
        plt.plot(self.X_train, self.predict(w, b, self.X_train))
        plt.show()
    

    def predict(self, w, b, x_pred):
        pred_fn = self.sigmoid(np.dot(w, x_pred)+b)
        return 1 if (pred_fn>=0.5) else 0
    

    def accuracy(self, w, b):
        tot = 0
        epsilon = 1e-1
        len = self.X_test.shape[0]
        for i in range(len):
            if ((self.predict(w, b, self.X_test[i]) == self.y_test[i])): tot+=1
        
        print("\n" + bcolors.BOLD + bcolors.UNDERLINE + bcolors.OKCYAN + "[+] Prediction Accuracy: " + str((tot/len)*100) + "%\n" + bcolors.ENDC)
        return (tot/len)*100

        






lr = LogisticRegression("iris-data.csv")
lr.change_label_names(["Iris-setossa","versicolor"], ["Iris-setosa","Iris-versicolor"])
lr.drop_label(['Iris-virginica'])
lr.label_encoding(["Iris-setosa","Iris-versicolor"])
lr.feature_scaling()
coeff, g, c = lr.Train()
print(coeff)
lr.accuracy(coeff['w'], coeff['b'])

# lr.feature_scaling()
# w, b = lr.Train()
# lr.accuracy(w, b, lr.X_train)