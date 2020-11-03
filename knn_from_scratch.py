import numpy as np

class KNN:

    def __init__(self, k=2, distance_metric="eucledian"):
        self.k =k 
        self.distance_metric = distance_metric
        self.X = None
        self.Y = None


    def fit(self, X, y):
        if len(X.shape) != 2 or len(y.shape) != 2:
            raise Exception("Input Shape must be 2D, your input shape is {}, try array.reshape(-1, 1)".format(X.shape))
        
        if self.k > self.X.shape[0]:
            raise Exception("Your nearest neighbour count is more than your input. Try add more data or decrease the number of neighbours")

        self.X = X
        self.y = y

        print("Model fitted to program")

    def predict(self, point):

        point = np.array(point)

        if len(point.shape) != len(self.X.shape):
            raise Exception("Your prediction array must be same shape with your Input Size.")
        if point.shape[1] != self.X.shape[1]:
            raise Exception("Your input size for prediction is not equal with your training data.",
                                 "Your input shape is {} and your prediction array shape is {}".format(self.X.shape, point.shape))

        if self.distance_metric == "eucledian":
           distances = np.sqrt(np.sum(np.square(self.X - point), axis=1))
        
        elif self.distance_metric == "manhattan":
            distances = np.sum(np.abs(self.X - y), axis=1)

        indices = np.argsort(distances)[:self.k]                    #get the indices of the closest points to given input
        near_labels = self.y[indices]                               #get the labels of closest points to given input
        counters =  np.unique(near_labels, return_counts=True)      #get the counts of every labels 
        labels = counters[0]
        values = counters[1] 
        max_ind_label = np.argsort(values)[-1]                      #get the index of label which has more closest points
        prediction = labels[max_ind_label]                          #return the label 

        return prediction 

        
        

if __name__ == "__main__":
    model = KNN(3, "eucledian")
    X = np.array([[4, 2], [3, 4], [1, 1], [2, 3]])
    y = np.array([[10], [2], [3], [10]])
    model.fit(X, y)
    print(model.predict([[2, 3]]))