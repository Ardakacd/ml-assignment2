import numpy as np

import matplotlib.pyplot as plt

import cv2


class Preprocessor:
    digits_of_interest = [2, 3, 8, 9]

    def read_mnist_images(self, filename):

        with open(filename, 'rb') as f:
            # Read the header information

            magic_number = int.from_bytes(f.read(4), 'big')

            num_images = int.from_bytes(f.read(4), 'big')

            num_rows = int.from_bytes(f.read(4), 'big')

            num_cols = int.from_bytes(f.read(4), 'big')

            # Read image data

            image_data = np.frombuffer(f.read(), dtype=np.uint8)

            image_data = image_data.reshape(num_images, num_rows, num_cols)

        return image_data

    def read_mnist_labels(self, filename):

        with open(filename, 'rb') as f:
            # Read the header information

            magic_number = int.from_bytes(f.read(4), 'big')

            num_labels = int.from_bytes(f.read(4), 'big')

            # Read label data

            label_data = np.frombuffer(f.read(), dtype=np.uint8)

        return label_data

    def preprocess_train_images(self, number=-1, visualize=False, with_hog_features=False):

        mnist_train_image_filename = 'train-images-idx3-ubyte'

        mnist_train_label_filename = 'train-labels-idx1-ubyte'

        flattened_train_images = self.read_mnist_images(mnist_train_image_filename)

        mnist_train_labels = self.read_mnist_labels(mnist_train_label_filename)

        filtered_indices = [i for i, label in enumerate(mnist_train_labels) if label in self.digits_of_interest]

        filtered_images = flattened_train_images[filtered_indices]

        if number == -1:

            y = np.array(

                [label if label in self.digits_of_interest else -1 for label in mnist_train_labels[filtered_indices]])

        else:

            y = np.where(mnist_train_labels[filtered_indices] == number, 1, -1)

        if visualize:
            self.visualize_image(filtered_images)

        if with_hog_features:
            filtered_train_images = self.compute_hog_features(filtered_images)
        else:
            filtered_train_images = filtered_images.reshape(filtered_images.shape[0], -1)

            filtered_train_images = filtered_train_images / 255

        return filtered_train_images, y

    def preprocess_test_images(self, number=-1, visualize=False, with_hog_features=False):

        mnist_test_image_filename = 't10k-images-idx3-ubyte'

        mnist_test_label_filename = 't10k-labels-idx1-ubyte'

        flattened_test_images = self.read_mnist_images(mnist_test_image_filename)

        mnist_test_labels = self.read_mnist_labels(mnist_test_label_filename)

        filtered_indices = [i for i, label in enumerate(mnist_test_labels) if label in self.digits_of_interest]

        filtered_images = flattened_test_images[filtered_indices]

        if number == -1:

            y = np.array(

                [label if label in self.digits_of_interest else -1 for label in mnist_test_labels[filtered_indices]])

        else:

            y = np.where(mnist_test_labels[filtered_indices] == number, 1, -1)

        if visualize:
            self.visualize_image(filtered_images)

        if with_hog_features:
            filtered_test_images = self.compute_hog_features(filtered_images)
        else:
            filtered_test_images = filtered_images.reshape(filtered_images.shape[0], -1)

            filtered_test_images = filtered_test_images / 255

        return filtered_test_images, y

    def compute_hog_features(self, images):

        hog_features = []

        hog = cv2.HOGDescriptor()

        for img in images:
            # Resize image to a fixed size (e.g., 64x64)

            img = cv2.resize(img, (64, 128))

            # Compute HOG features for the image

            features = hog.compute(img)

            # Reshape the feature vector and append to the list

            hog_features.append(features.flatten())

        return np.array(hog_features)

    def visualize_image(self, images):

        num_images_to_display = 10

        for i in range(num_images_to_display):
            plt.subplot(1, num_images_to_display, i + 1)

            plt.imshow(images[i], cmap='gray')

            plt.axis('off')

        plt.show(block=False)

        # this will show the plot for 5 seconds

        plt.pause(5)

        plt.close()
