import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class KNN:

    def __init__(self, k=2, distance_metric="eucledian"):
        
        if distance_metric != "eucledian" and distance_metric != "manhattan":
            raise Exception("Distance metric must be eucledian or manhattan")

        self.k =k 
        self.distance_metric = distance_metric
        self.X = None
        self.Y = None


    def fit(self, X, y):

        if len(X.shape) != 2 or len(y.shape) != 2:
            raise Exception("Input Shape must be 2D, your input shape is {}, try array.reshape(-1, 1)".format(X.shape))

        self.X = X
        self.y = y
        

        if self.k > self.X.shape[0]:
            raise exception("your nearest neighbour count is more than your input. try add more data or decrease the number of neighbours")

        print("Model fitted to program")

    def predict(self, prediction_X):

        prediction_X = np.array(prediction_X)
        predictions = []

        for point in prediction_X:

            try:
                point = point.reshape((1, self.X.shape[1]))
            except:
                raise Exception("your prediction test data is not same size with your input data."
                                "Your training data has {} features and your test data shape is {}".format(X_train.shape[1],point.shape))

            if len(point.shape) != len(self.X.shape):
                raise Exception("Your prediction array must be same shape with your Input Size.")

            if point.shape[1] != self.X.shape[1]:
                print(point.shape[0], self.X.shape[1])
                raise Exception("Your input size for prediction is not equal with your training data.",
                                    "Your input shape is {} and your prediction array shape is {}".format(self.X.shape, prediction_X.shape))

            if self.distance_metric == "eucledian":
                distances = np.sqrt(np.sum(np.square(self.X - point), axis=1))
            
            elif self.distance_metric == "manhattan":
                distances = np.sum(np.abs(self.X - point), axis=1)

            indices = np.argsort(distances)[:self.k]                    #get the indices of the closest points to given input
            near_labels = self.y[indices]                               #get the labels of closest points to given input
            counters =  np.unique(near_labels, return_counts=True)      #get the counts of every labels 
            labels = counters[0]
            values = counters[1] 
            max_ind_label = np.argmax(values)                           #get the index of label which has more closest points
            prediction = labels[max_ind_label]    
            predictions.append(prediction)                              #return the label 

        return predictions 

        
        

if __name__ == "__main__":
    model = KNN(10, "eucledian")
    data = load_iris()
    X = data.data
    y = data.target.reshape(-1, 1) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    predictions = np.array(model.predict(X_test)).reshape(-1, 1)
    is_predictions_true = np.array(predictions == y_test)
    accuracy  = np.sum(is_predictions_true) / len(y_test)
    print("Accuracy is ", accuracy)
