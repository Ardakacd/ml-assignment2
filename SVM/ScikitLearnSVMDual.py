from sklearn.model_selection import  GridSearchCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from SVM.Preprocessor import Preprocessor


class ScikitLearnSVMDual:
    def __init__(self, kernel='rbf', param_grid={'estimator__C': [0.1, 1, 10], 'estimator__gamma': [0.1, 1, 10]}):
        self.kernel = kernel
        self.param_grid = param_grid
        self.classifier = OneVsRestClassifier(SVC(kernel=kernel, max_iter=100000))

    def train(self, X_train, y_train):
        grid_search = GridSearchCV(self.classifier, self.param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.best_C = grid_search.best_params_['estimator__C']
        self.best_gamma = grid_search.best_params_['estimator__gamma']
        self.classifier = grid_search.best_estimator_

    def evaluate(self, X_train, y_train, X_test, y_test):
        train_accuracy = accuracy_score(y_train, self.classifier.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.classifier.predict(X_test))
        return train_accuracy, test_accuracy


preprocessor = Preprocessor()
X_train, y_train = preprocessor.preprocess_train_images(-1)
X_test, y_test = preprocessor.preprocess_test_images(-1)

svm = ScikitLearnSVMDual(kernel='rbf')
svm.train(X_train, y_train)
train_accuracy, test_accuracy = svm.evaluate(X_train, y_train, X_test, y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
