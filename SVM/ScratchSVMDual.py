import time
import random

import numpy as np
from cvxopt import matrix, solvers

from Preprocessor import Preprocessor


class ScratchSVMDual:

    # possible values for 3-fold validation
    C = [0.1, 1, 100]
    gamma = [0.001, 0.1, 0.01]


    # implementation of 3-fold validation
    # by using this function we determined the C and gamma value
    def three_fold_validation(self):
        preprocessor = Preprocessor()
        overall_accuracies = []
        for ind in range(len(self.C)):
            accuracies = []
            for digit in Preprocessor.digits_of_interest:
                X, y = preprocessor.preprocess_train_images(digit)
                fold_size = len(X) // 3
                fold_accuracy = []
                for fold in range(3):
                    test_indices = range(fold * fold_size, (fold + 1) * fold_size)
                    train_indices = [i for i in range(len(X)) if i not in test_indices]

                    X_train, y_train = X[train_indices], y[train_indices]
                    X_test, y_test = X[test_indices], y[test_indices]

                    self.train(X_train, y_train, self.C[ind], self.gamma[ind])
                    y_pred = self.test(X_test, y_test,self.gamma[ind])
                    fold_accuracy.append(np.mean(y_pred == y_test))

                accuracies.append(np.mean(fold_accuracy))
            overall_accuracies.append(accuracies)
        return overall_accuracies

    def rbf_kernel(self, X, gamma):
        # Compute the pairwise squared Euclidean distances
        distances_sq = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
        # Compute RBF kernel matrix
        return np.exp(-gamma * distances_sq)

    def train(self, X, y, C, gamma):
        n_samples, n_features = X.shape

        # Gram matrix
        K = self.rbf_kernel(X,gamma)

        P = matrix(np.outer(y, y) * K)
        
        q = matrix(np.ones(n_samples) * -1)
        
        A = matrix(y, (1, n_samples), 'd')
        
        b = matrix(0.0)
        
        G = matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
        
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
       
        solution = solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(solution['x'])

        # Support vectors
        sv = alpha > 1e-5
        self.alpha = alpha[sv]
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]

        self.b = np.mean(y[sv] - self.predict_value(self.support_vectors,gamma))

    def predict_value(self, X, gamma):
        distances_sq = np.sum(X ** 2, axis=1, keepdims=True) - 2 * np.dot(X, self.support_vectors.T) + np.sum(
            self.support_vectors ** 2, axis=1)
        # Compute RBF kernel matrix
        k_matrix = np.exp(-gamma * distances_sq)

        predicted_values = np.dot(k_matrix, self.alpha * self.support_vector_labels) + self.b if hasattr(self,
                                                                                                         'b') else 0
        return predicted_values

    def test(self, X_test, y_test, gamma):
        y_pred = self.predict_value(X_test, gamma)
        for ind in range(len(y_pred)):
            if y_pred[ind] > 0:
                y_pred[ind] = 1
            else:
                y_pred[ind] = -1
        
        print(np.mean(y_pred == y_test))
        return y_pred





# after 3 fold C is selected as 100 and gamma is selected as 0.1
overall_y_train = []
overall_y_test = []

preprocessor = Preprocessor()
s = ScratchSVMDual()

starting_time = time.time()



for digit in Preprocessor.digits_of_interest:
    X_train, y_train = preprocessor.preprocess_train_images(digit)
    X_test, y_test = preprocessor.preprocess_test_images(digit)
    s.train(X_train, y_train, 100, 0.1)
    overall_y_train.append(s.test(X_train,y_train, 0.1))
    overall_y_test.append(s.test(X_test,y_test, 0.1))

print("training time:", time.time() - starting_time, "seconds")

result_y_train = []

for ind in range(len(overall_y_train[0])):
    temp = []
    if overall_y_train[0][ind] == 1:
        temp.append(2)

    if overall_y_train[1][ind] == 1:
        temp.append(3)

    if overall_y_train[2][ind] == 1:
        temp.append(8)

    if overall_y_train[3][ind] == 1:
        temp.append(9)

    if len(temp) == 0:
        temp = [2, 3, 8, 9]

    result_y_train.append(random.choice(temp))

result_y_test = []

for ind in range(len(overall_y_test[0])):
    temp = []
    if overall_y_test[0][ind] == 1:
        temp.append(2)

    if overall_y_test[1][ind] == 1:
        temp.append(3)

    if overall_y_test[2][ind] == 1:
        temp.append(8)

    if overall_y_test[3][ind] == 1:
        temp.append(9)

    if len(temp) == 0:
        temp = [2, 3, 8, 9]

    result_y_test.append(random.choice(temp))

_, y_test = preprocessor.preprocess_test_images()
_, y_train = preprocessor.preprocess_train_images()
print(np.mean(result_y_train == y_train))
print(np.mean(result_y_test == y_test))