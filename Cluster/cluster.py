from Preprocessor import Preprocessor
import numpy as np
import random
import matplotlib.pyplot as plt


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
        # Randomly initialize cluster centroids
        centroids = [X[i] for i in random.sample(range(len(X)), self.num_clusters)]

        for _ in range(self.max_iterations):
            # Assign each data point to the nearest centroid
            distances = np.zeros((X.shape[0], self.num_clusters))
            for i, centroid in enumerate(centroids):
                for j, data_point in enumerate(X):
                    if self.distance_type == 'euclidean':
                        distances[j, i] = euclidean_distance(data_point, centroid)
                    elif self.distance_type == 'cosine':
                        distances[j, i] = cosine_distance(data_point, centroid)
            cluster_assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(self.num_clusters):
                cluster_points = X[cluster_assignments == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.cluster_assignments = cluster_assignments

    def predict(self, X):
        distances = np.zeros((X.shape[0], self.num_clusters))
        for i, centroid in enumerate(self.centroids):
            for j, data_point in enumerate(X):
                if self.distance_type == 'euclidean':
                    distances[j, i] = euclidean_distance(data_point, centroid)
                elif self.distance_type == 'cosine':
                    distances[j, i] = cosine_distance(data_point, centroid)
        return np.argmin(distances, axis=1)

preprocessor = Preprocessor()
X_train, y_train = preprocessor.preprocess_train_images(-1)

print("XTrain shape: ")  
print(X_train.shape)  


kmeans = KMeans(num_clusters=4, max_iterations=200, distance_type='euclidean')
kmeans.fit(X_train)
cluster_assignments = kmeans.cluster_assignments
centroids = kmeans.centroids
print(cluster_assignments)
print(centroids)
print(centroids[0].shape)
plot_centroids(centroids=centroids) 

