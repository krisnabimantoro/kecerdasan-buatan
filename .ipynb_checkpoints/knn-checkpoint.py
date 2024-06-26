import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Generate random data
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Visualize decision boundary with dispersion
h = .02  # Step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Calculate dispersion
dispersion = np.zeros_like(Z, dtype=float)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, -1)
        distances, _ = knn.kneighbors(point, k)
        dispersion[i, j] = np.mean(distances)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# Overlay dispersion as contour lines
plt.contour(xx, yy, dispersion, levels=np.linspace(dispersion.min(), dispersion.max(), 10), cmap='Greys', alpha=0.5)

plt.title("KNN Classification with Dispersion")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
