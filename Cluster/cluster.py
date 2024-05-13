from Preprocessor import Preprocessor
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def cosine_distance(x1, x2):
    return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def plot_centroids(centroids, image_shape=(28, 28)):
    _, axes = plt.subplots(1, len(centroids), figsize=(12, 3))
    for ax, centroid in zip(axes, centroids):
        ax.imshow(centroid.reshape(image_shape), cmap="gray")
        ax.axis("off")
    plt.show()

class KMeans:
    def __init__(self, num_clusters=4, max_iterations=100, distance_type='euclidean'):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.distance_type = distance_type

    def fit(self, X):
        centroids = [X[i] for i in random.sample(range(len(X)), self.num_clusters)]

        for _ in range(self.max_iterations):
            distances = np.zeros((X.shape[0], self.num_clusters))
            for i, centroid in enumerate(centroids):
                for j, data_point in enumerate(X):
                    if self.distance_type == 'euclidean':
                        distances[j, i] = euclidean_distance(data_point, centroid)
                    elif self.distance_type == 'cosine':
                        distances[j, i] = cosine_distance(data_point, centroid)
            cluster_assignments = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for i in range(self.num_clusters):
                cluster_points = X[cluster_assignments == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[i] = centroids[i]
            centroids = new_centroids

        self.centroids = centroids
        self.cluster_assignments = cluster_assignments
        
        self.sse = 0
        for i, centroid in enumerate(self.centroids):
            cluster_points = X[self.cluster_assignments == i]
            self.sse += np.sum((cluster_points - centroid) ** 2)

    def predict(self, X):
        distances = np.zeros((X.shape[0], self.num_clusters))
        for i, centroid in enumerate(self.centroids):
            for j, data_point in enumerate(X):
                if self.distance_type == 'euclidean':
                    distances[j, i] = euclidean_distance(data_point, centroid)
                elif self.distance_type == 'cosine':
                    distances[j, i] = cosine_distance(data_point, centroid)
        return np.argmin(distances, axis=1)
    
def assign_labels_to_clusters(labels, true_labels, k):
    assigned_labels = {}
    for cluster_idx in range(k):
        cluster_labels = true_labels[labels == cluster_idx]
        label_counts = Counter(cluster_labels)
        most_common_label = label_counts.most_common(1)[0][0]
        assigned_labels[cluster_idx] = most_common_label
    return assigned_labels

def calculate_accuracy(true_labels, predicted_labels, k):
    correct = 0
    assigned_labels = assign_labels_to_clusters(predicted_labels, true_labels, k)
    for i in range(true_labels.shape[0]):
        if assigned_labels[predicted_labels[i]] == true_labels[i]:
            correct += 1
    return correct / true_labels.shape[0]

preprocessor = Preprocessor()

X_train, y_train = preprocessor.preprocess_train_images(-1, normalize=True)
X_test, y_test = preprocessor.preprocess_test_images(-1, normalize=True)

kmeans = KMeans(distance_type='euclidean')
kmeans.fit(X_train)
cluster_assignments = kmeans.cluster_assignments
centroids = kmeans.centroids

sse = kmeans.sse
print("Sum of Squared Errors (SSE) for Euclidean Distance:", sse)

predicted_cluster_assignments = kmeans.predict(X_test)
accuracy = calculate_accuracy(y_test, predicted_cluster_assignments, kmeans.num_clusters)
print("Accuracy for Euclidean Distance:", accuracy)

print()

kmeans = KMeans(distance_type='cosine')
kmeans.fit(X_train)
cluster_assignments = kmeans.cluster_assignments
centroids = kmeans.centroids

sse = kmeans.sse
print("Sum of Squared Errors (SSE) for Cosine Distance:", sse)

predicted_cluster_assignments = kmeans.predict(X_test)
accuracy = calculate_accuracy(y_test, predicted_cluster_assignments, kmeans.num_clusters)
print("Accuracy for Cosine Distance:", accuracy)