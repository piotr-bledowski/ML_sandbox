import numpy as np

class Perceptron:
    def __init__(self):
        self.theta = None
        self.theta_0 = None

    def fit(self, X_train: np.ndarray, y_train: np.array, T: int):
        """
        Train the perceptron on some data

        Parameters:
            X_train (np.ndarray): Features (2d array of floats)
            y_train (np.array): Target values (2d array of ints)
            T (int): Number of iterations (how many times should the algorithm evaluate all the datapoints)

        Returns:
            None: it updates the class fields
        """
        # dimension of theta should be equal to the dimension of feature vectors in X
        self.theta = np.zeros(X_train.shape[1])
        self.theta_0 = 0

        n = X_train.shape[0] # Number of datapoints

        y = np.where(y_train > 0, 1, 0)

        for t in range(T):
            for i in range(n):
                # the condition below is satisfied iff the prediction of the current perceptron state and the actual value mismatch
                yp = self.predict(X_train[i])
                update = 0.1 * (y[i] - yp)
                self.theta += update * X_train[i]
                self.theta_0 += update

    def predict(self, x: np.array) -> int:
        """
        Predict the class of a given datapoint x

        Parameters:
            x (np.array): a new datapoint

        Returns:
            y (int): either 1 or -1 for binary classification problem
        """
        return 1 if np.dot(self.theta, x) + self.theta_0 > 0 else 0
