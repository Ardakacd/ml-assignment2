import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

from utils.Preprocessor import Preprocessor


class ScratchSVMDual:
    C = [0.1, 1, 10]
    sigma = [0.1, 1, 10]

    def train(self, X_train, y_train, C, sigma):
        row, col = X_train.shape
        y_train = y_train.reshape(-1, 1) * 1.
        P = cvxopt_matrix(np.matmul(y_train, y_train.T) * self.gaussian_kernel(X_train, X_train, sigma))
        q = cvxopt_matrix(-np.ones((row, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(row) * -1, np.eye(row))))
        h = cvxopt_matrix(np.hstack((np.zeros(row), np.ones(row) * C)))
        A = cvxopt_matrix(y_train.reshape(1, -1))
        c = cvxopt_matrix(np.zeros(1))

        sol = cvxopt_solvers.qp(P, q, G, h, A, c)
        alphas = np.array(sol['x'])

        w = ((y_train * alphas).T @ X_train).reshape(-1, 1)
        S = (alphas > 1e-4).flatten()
        b = y_train[S] - np.dot(X_train[S], w)

        print('Alphas = ', alphas[alphas > 1e-4])
        print('w = ', w.flatten())
        print('b = ', b[0])

        return w.flatten(), b[0], alphas[alphas > 1e-4]

    def test(self, b, alphas, y_train, y_test, X_train, X_test, sigma):
        kernel_matrix = self.gaussian_kernel(X_train, X_test,sigma)
        y_pred = np.sign(np.dot((alphas * y_train).T, kernel_matrix) + b)
        accuracy = np.mean(y_pred == y_test)
        return accuracy

    def gaussian_kernel(self, f, s, sigma):
        n = f.shape[0]
        m = s.shape[0]
        xx = np.dot(np.sum(np.power(f, 2), 1).reshape(n, 1), np.ones((1, m)))
        zz = np.dot(np.sum(np.power(s, 2), 1).reshape(m, 1), np.ones((1, n)))
        return np.exp(-(xx + zz.T - 2 * np.dot(f, s.T)) / (2 * sigma ** 2))


preprocessor = Preprocessor()
X_train, y_train = preprocessor.preprocess_train_images(2)
X_test, y_test = preprocessor.preprocess_test_images(2)

s = ScratchSVMDual()

w, b, alphas = s.train(X_train, y_train, 1, 1)

print(s.test(b, alphas, y_train, y_test, X_train, X_test,1))
