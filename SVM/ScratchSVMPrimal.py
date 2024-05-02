import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

from SVM.Preprocessor import Preprocessor


class ScratchSVMPrimal:
    C = [0.1, 1, 10]
    sigma = [0.1, 1, 10]

    def train(self, X_train, y_train, C):
        row, col = X_train.shape
        y_train = y_train.astype(float)
        Q = np.zeros((col + 1 + row, col + 1 + row))
        Q[:col, :col] = np.eye(col)
        p = np.zeros(col + 1 + row)
        p[-row:] = C
        G = np.zeros((2 * row, col + 1 + row))
        G[:row, :col] = -np.diag(y_train) @ X_train
        G[:row, col] = -y_train
        G[:row, -row:] = -np.eye(row)
        G[row:, -row:] = -np.eye(row)
        h = -np.ones(2 * row)
        h[:row] = -1
        sol = cvxopt_solvers.qp(cvxopt_matrix(Q), cvxopt_matrix(p), cvxopt_matrix(G), cvxopt_matrix(h))
        w = np.array(sol['x'])[:col]
        b = np.array(sol['x'])[col]

        print(w)
        print(b)

        return w, b

    def test(self,X_test, w, b, y_test):
        y_pred = np.sign(np.dot(X_test, w) + b)
        return np.mean(y_pred == y_test)


preprocessor = Preprocessor()
X_train, y_train = preprocessor.preprocess_train_images(2)
X_test, y_test = preprocessor.preprocess_test_images(2)

s = ScratchSVMPrimal()

w, b = s.train(X_train, y_train, 1)

print(s.test(X_test, w, b, y_test))
