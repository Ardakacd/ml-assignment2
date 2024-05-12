import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

from Preprocessor import Preprocessor


class ScratchSVMPrimal:
   #C = [0.1, 1, 10]

   def train(self, X_train, y_train, C):
       row, col = X_train.shape
       y_train = y_train.astype(float)

       Q = np.zeros((col + 1 + row, col + 1 + row))  # b, w, and epsilon
       Q[1:col+1, 1:col+1] = np.eye(col)
       p = np.hstack([np.zeros(col + 1), C * np.ones(row)])
       A = np.zeros((2 * row, col + 1 + row))
       c = np.zeros(2 * row)
       for i in range(row):
           A[i, 0] = y_train[i]
           A[i, 1:col+1] = y_train[i] * X_train[i]
           A[i, col+1+i] = 1
           c[i] = 1
           A[row+i, col+1+i] = 1

       sol = cvxopt_solvers.qp(cvxopt_matrix(Q), cvxopt_matrix(p), cvxopt_matrix(-A), cvxopt_matrix(-c))
       b = sol["x"][0]
       w = sol["x"][1:col+1]

       return w, b

   def test(self,X_test, w, b, y_test):
       y_pred = []
       for ind in range(len(X_test)):
           v = np.dot(X_test[ind], w) + b > 0
           if v > 0:
               y_pred.append(1)
           else:
               y_pred.append(-1)
       y_pred = np.array(y_pred)
       return np.mean(y_pred == y_test)


preprocessor = Preprocessor()
X_train, y_train = preprocessor.preprocess_train_images(3)
X_test, y_test = preprocessor.preprocess_test_images(3)

s = ScratchSVMPrimal()

w, b = s.train(X_train, y_train, 100)

print(s.test(X_test, w, b, y_test))