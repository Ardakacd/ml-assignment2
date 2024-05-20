import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import random
from Preprocessor import Preprocessor
import time 


class ScratchSVMPrimal:

  # possible values for 3-fold validation
   C = [1, 10, 100]

    # implementation of 3-fold validation
    # by using this function we determined the C value
   def three_fold_validation(self):
        preprocessor = Preprocessor()
        overall_accuracies = []
        for c in self.C:
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

                    w, b = self.train(X_train, y_train, c)
                    y_pred = self.test(X_test, w, b, y_test)
                    fold_accuracy.append(np.mean(y_pred == y_test))

                accuracies.append(np.mean(fold_accuracy))
            overall_accuracies.append(accuracies)
        return overall_accuracies

   def train(self, X_train, y_train, C):
       row, col = X_train.shape
       y_train = y_train.astype(float)

       p = np.hstack([np.zeros(col + 1), np.ones(row) * C])

       A = np.zeros((2 * row, col + 1 + row))
       for i in range(row):
           A[i, 0] = y_train[i]
           A[i, 1:col+1] = y_train[i] * X_train[i]
           A[i, col+i+1] = 1
           A[row+i, col+i+1] = 1

       Q = np.zeros((col + row + 1, col + row + 1)) 
       Q[1:col+1, 1:col+1] = np.eye(col)

       c = np.zeros(2 * row)
       for i in range(row):
           c[i] = 1
       
       sol = cvxopt_solvers.qp(cvxopt_matrix(Q), cvxopt_matrix(p), cvxopt_matrix(-A), cvxopt_matrix(-c))
       b = sol["x"][0]
       w = sol["x"][1:col+1]

       return w, b

   def test(self,X_test, w, b, y_test):
        y_pred = []
        for ind in range(len(X_test)):
           # if sign is positive then add 1 otherwise add -1
           v = np.dot(X_test[ind], w) + b > 0
           if v > 0:
               y_pred.append(1)
           else:
               y_pred.append(-1)
        print(np.mean(y_pred == y_test))
        return np.array(y_pred)
       


preprocessor = Preprocessor()
s = ScratchSVMPrimal()


# after 3 fold 100 is selected as C
overall_y_train = []
overall_y_test = []

starting_time = time.time()

for digit in Preprocessor.digits_of_interest:
    X_train, y_train = preprocessor.preprocess_train_images(digit)
    X_test, y_test = preprocessor.preprocess_test_images(digit)
    w, b = s.train(X_train, y_train, 100)
    overall_y_test.append(s.test(X_test, w, b, y_test))
    overall_y_train.append(s.test(X_train, w, b, y_train))

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
        temp = [2,3,8,9]

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
        temp = [2,3,8,9]

    result_y_test.append(random.choice(temp))
   
        


_, y_test = preprocessor.preprocess_test_images()
_, y_train = preprocessor.preprocess_train_images()
print(np.mean(result_y_train == y_train))
print(np.mean(result_y_test == y_test))