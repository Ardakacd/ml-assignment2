from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

from Assignment2.SVM.Preprocessor import Preprocessor


class ScikitLearnSVMPrimal:
    def __init__(self, param_grid={'estimator__C': [0.1, 1, 10]}):
        self.param_grid = param_grid
        self.classifier = OneVsRestClassifier(LinearSVC(max_iter=100000,dual=False))

    def train(self, X_train, y_train):
        grid_search = GridSearchCV(self.classifier, self.param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.best_C = grid_search.best_params_['estimator__C']
        self.classifier = grid_search.best_estimator_

    def evaluate(self, X_train, y_train, X_test, y_test):
        train_accuracy = accuracy_score(y_train, self.classifier.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.classifier.predict(X_test))
        return train_accuracy, test_accuracy


preprocessor = Preprocessor()

X_train,y_train = preprocessor.preprocess_train_images(-1)


X_test,y_test = preprocessor.preprocess_test_images(-1)

svm = ScikitLearnSVMPrimal()

svm.train(X_train, y_train)

train_accuracy, test_accuracy = svm.evaluate(X_train, y_train, X_test, y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")



# Training accuracy: 0.9552
# Test accuracy: 0.9439
