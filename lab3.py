import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder

# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Define a KNN classifier class
class KNN:
    def __init__(self, k=5):
        self.k = k
    
    # Fit the model to the training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # Predict the class labels for the test data
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    # Helper function to predict the class label for a single data point
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.argmax(np.bincount(k_nearest_labels))
        return most_common

# Read in the Iris dataset from a CSV file
data = pd.read_csv('Iris.csv')

# Select two columns to use for classification
selected_columns = ['PetalLengthCm', 'PetalWidthCm']

# Extract the selected columns and the class labels
X = data[selected_columns].values
y = data['Species'].values

# Convert the class labels to integer labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and test sets
np.random.seed(42)
indices = np.random.permutation(len(X))
X_train = X[indices[:int(0.8*len(X))]]
y_train = y[indices[:int(0.8*len(y))]]
X_test = X[indices[int(0.8*len(X)):]]
y_test = y[indices[int(0.8*len(y)):]]

# Train a KNN classifier on the training data
knn = KNN(k=5)
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Print the predicted and actual class labels for each test data point
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)
for i in range(len(y_pred)):
    print('Predicted: {}, Actual: {}'.format(y_pred_labels[i], y_test_labels[i]))

# Calculate the error rate
error = np.mean(y_pred != y_test)
print('Error rate:', error)

# Plot the test data with the predicted class labels
colors = [plt.cm.viridis(each) for each in np.linspace(0, 1, len(np.unique(y_pred)))]
patches = [mpatches.Patch(color=colors[i], label=f'{y_pred_labels[i]}') for i in range(len(np.unique(y_pred)))]
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
plt.xlabel('Petal Length(mm)')
plt.ylabel('Petal Width(mm)')
plt.title('Fishers Iris Data')
plt.legend(handles=patches)
plt.show()