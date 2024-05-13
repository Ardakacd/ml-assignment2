import sys

sys.path.append("../")

from utils.Preprocessor import Preprocessor

preprocessor = Preprocessor()
X_train, y_train = preprocessor.preprocess_train_images(-1,False,True)
X_test, y_test = preprocessor.preprocess_test_images(-1,False,True)