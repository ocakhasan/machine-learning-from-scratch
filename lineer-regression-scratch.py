import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

class LineerRegression():

    def fit(self, X, y, learning_rate=0.01):

        theta_dimension = (1 + X.shape[1], 1)
        self.THETA = np.zeros(theta_dimension, dtype=np.float32)
        self.learning_rate = learning_rate

        xzeros = np.ones([X.shape[0], 1], dtype=X.dtype)                        #Add meaningless 0 data for some convention
        X = np.concatenate((xzeros, X), axis=1)

        for x_m, y_m in zip(X, y):
    
            hypothesis_x = self.hypothesis(x_m, self.THETA)                              #calculate hypothesis
            loss = self.loss_function(hypothesis_x, y_m)                                 #calculate loss function
            THETA = self.update_parameters(self.THETA, hypothesis_x,                    
                                            x_m, y_m, self.learning_rate)                 #Update THETA parameters

        print("Training is done")
        print("The parameters are \n", THETA)
    
    def predict(self, X_test):

        xzeros = np.ones([X_test.shape[0], 1], dtype=X_test.dtype)                      #Add meanningless 0 data again
        X_test = np.concatenate((xzeros, X_test), axis=1)
        predictions = []

        for x_z in X_test:
            prediction = self.hypothesis(x_z, self.THETA)
            predictions.append(prediction)

        print("Predictions are done")
        return np.array(predictions)


    def hypothesis(self, x, θ):
            hypothesis = np.matmul(θ.T, x)
            return hypothesis

    def loss_function(self, h_x, y):
            loss_error = 1 /2 *(h_x - y) *(h_x - y)
            return loss_error

    def update_parameters(self, THETA, hypothesis, x, y, learning_rate):
            
            x = np.reshape(x, THETA.shape)
            Updated_THETA = THETA - learning_rate *(hypothesis - y) *x
            return Updated_THETA

def main():
    X, y = make_regression(n_samples = 250, n_features=3, noise=20, random_state=0, bias=50)                #Create some dummy regression data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print("Data is created")
    model  = LineerRegression()                             
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(mean_squared_error(predictions, y_test))

    


if __name__ == "__main__":
    main()