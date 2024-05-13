import time

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from utils.Preprocessor import Preprocessor
import matplotlib.pyplot as plt
import numpy as np


class ScikitLearnSVMDual:
    def __init__(self, kernel='rbf', param_grid={'estimator__C': [1, 10,100], 'estimator__gamma': [0.01, 0.1, 1]}):
        self.kernel = kernel
        self.param_grid = param_grid
        self.classifier = OneVsRestClassifier(SVC(kernel=kernel,max_iter=1000))

    def train(self, X_train, y_train):
        grid_search = GridSearchCV(self.classifier, self.param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        self.best_C = grid_search.best_params_['estimator__C']
        self.best_gamma = grid_search.best_params_['estimator__gamma']
        self.classifier = grid_search.best_estimator_

    def evaluate(self, X_train, y_train, X_test, y_test):
        train_accuracy = accuracy_score(y_train, self.classifier.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.classifier.predict(X_test))
        return train_accuracy, test_accuracy

    def visualize_support_vectors(self, X_train, y_train, num_support_vectors_to_display=8):

        self.train(X_train, y_train)

        support_vectors_indices = self.classifier.estimators_[0].support_

        if len(support_vectors_indices) > num_support_vectors_to_display:
            selected_indices = np.random.choice(support_vectors_indices, size=num_support_vectors_to_display,
                                                replace=False)
        else:
            selected_indices = support_vectors_indices

        print(selected_indices)
        print(len(selected_indices))

        X_train = X_train * 255

        # Visualize data and selected support vectors
        plt.figure(figsize=(10, 6))
        for i, index in enumerate(selected_indices):
            plt.subplot(2, num_support_vectors_to_display // 2, i + 1)
            plt.imshow(X_train[index].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title(f'Support Vector {i + 1}')
            plt.xticks(())
            plt.yticks(())
        plt.tight_layout()
        plt.show()


preprocessor = Preprocessor()
X_train, y_train = preprocessor.preprocess_train_images(-1,False,True)
X_test, y_test = preprocessor.preprocess_test_images(-1,False,True)

svm = ScikitLearnSVMDual(kernel='rbf')
starting_time = time.time()
svm.train(X_train, y_train)
print("training time:", time.time() - starting_time, "seconds")
train_accuracy, test_accuracy = svm.evaluate(X_train, y_train, X_test, y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
