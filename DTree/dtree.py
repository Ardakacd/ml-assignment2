import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# seed is assigned for any randomized process (splitting, sampling, etc)
# we want to produce same results while developing and initiali testing
seed = 462  
np.random.seed(seed)

# load and shuffle data
data_file = 'data/wdbc.data'
raw_data = np.genfromtxt(data_file, delimiter=',', dtype=None, encoding=None)
np.random.shuffle(raw_data)

# assign -1 and 1 to target classes since we will be training linear classifier on it later on
classes = np.array([-1 if row[1] == 'M' else 1 if row[1] == 'B' else None for row in raw_data]) # M : -1 , B : 1

# feature matrix = don't take id and target classes in it
feature_matrix = np.array([list(row)[2:] for row in raw_data])

# train-test splitting
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, classes, test_size=0.2, random_state=462)

"""
PART 1
"""

for max_depth in range(3,9,2):
    # depth tuning and training the classifier
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)

    # visualize the generated tree
    plt.figure(figsize=(10,8))
    plot_tree(clf, filled=True)
    plt.title(f"DT with max depth of {max_depth}")
    plt.show()

# test the generated tree on test set
y_pred = clf.predict(X_test)

#full_tree_depth = clf.tree_.max_depth              # we've observed that full tree has depth of seven
#print("Depth of the full tree:", full_tree_depth)

# obtained accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree accuracy:", accuracy)

"""
PART 3
can observe results obtained from feature selection
"""

# feature importances
feature_importances = clf.feature_importances_

# Get indices of features sorted by importance
indices = feature_importances.argsort()[::-1]

# holds indices of most important 20 feature (will train linear classifier on first 5-10-15-20)
first_20_feature_indices = []
importance_scores = []

for f in range(20):          # 5-10-15-20
    #print("%d. Feature %d (%f)" % (f + 1, indices[f], feature_importances[indices[f]]))
    importance_scores.append(feature_importances[indices[f]])
    first_20_feature_indices.append(indices[f])

#print(first_20_feature_indices)
#print(indice_score_pairs)
plt.figure(figsize=(10, 6))
plt.bar(first_20_feature_indices, importance_scores, color='blue')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.title('Most important 20 features in DT (ordered by their index in x-axis)')
plt.xticks(first_20_feature_indices)
plt.show()


# we have selected logistic regression as our linear classifier
def train_lr(X_train, X_test, y_train, y_test, features):
    X_train_selected = X_train[:, features]
    X_test_selected = X_test[:, features]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)
    
    y_pred = lr.predict(X_test_scaled)    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

print("\nLinear Classifier Performance Stats :")
print(f"Accuracy for 5 features : {train_lr(X_train, X_test, y_train, y_test, first_20_feature_indices[:5])}")
print(f"Accuracy for 10 features : {train_lr(X_train, X_test, y_train, y_test, first_20_feature_indices[:10])}")
print(f"Accuracy for 15 features : {train_lr(X_train, X_test, y_train, y_test, first_20_feature_indices[:15])}")
print(f"Accuracy for 20 features : {train_lr(X_train, X_test, y_train, y_test, first_20_feature_indices[:20])}")


"""
PART 4
"""

"""
we can modify the number of trees in the forest by changing 
"n_estimators" parameter (default value is 100) in random forest classifier
"""

def train_random_forest(X_train, X_test, y_train, y_test):
    # random forest classifier of scikit learn
    #print("\nRandom Forest Accuracy Values : ")
    accuracy_vals = []
    tree_sizes = []
    print("\nRandom Forest Stats :")
    for num_trees in range(10,110,10):
        rforest = RandomForestClassifier(n_estimators=num_trees)
        rforest.fit(X_train, y_train)
        y_pred = rforest.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_vals.append(accuracy)
        tree_sizes.append(num_trees)
        print(f"Accuracy for {num_trees} trees : ", accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(tree_sizes, accuracy_vals, marker='o', color='b', linestyle='-')
    plt.xlabel('Tree Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Tree Size in Random Forests')
    plt.grid(True)
    plt.show()
    print(f"Average accuracy of random forest : {sum(accuracy_vals) / len(accuracy_vals)}")

train_random_forest(X_train, X_test, y_train, y_test)