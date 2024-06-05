import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess fingerprint images
def load_images_from_folder(folder, img_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
    return np.array(images)

folder_path = r'C:\Kuliah\Semester 4\kecerdasan buatan\fingerprint'  # Replace with the actual path
images = load_images_from_folder(folder_path)
images = images.astype('float32') / 255.0  # Normalize to [0, 1]
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

# Reshape images to add channel dimension
train_images = np.expand_dims(train_images, axis=-1)
val_images = np.expand_dims(val_images, axis=-1)

# Build the CCAE network
def build_ccae(input_shape=(128, 128, 1)):
    input_img = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder

input_shape = (128, 128, 1)
autoencoder = build_ccae(input_shape)
autoencoder.summary()

# Train the autoencoder
history = autoencoder.fit(train_images, train_images,
                          epochs=50,
                          batch_size=32,
                          shuffle=True,
                          validation_data=(val_images, val_images))

# Get encoded representations
encoder = models.Model(autoencoder.input, autoencoder.get_layer(index=6).output)
encoded_images = encoder.predict(train_images)

# Flatten the encoded images
encoded_images_flattened = encoded_images.reshape((encoded_images.shape[0], -1))

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
encoded_images_pca = pca.fit_transform(encoded_images_flattened)

# Apply clustering
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans_labels = kmeans.fit_predict(encoded_images_pca)

agglo = AgglomerativeClustering(n_clusters=10)
agglo_labels = agglo.fit_predict(encoded_images_pca)

# Create the GUI using Matplotlib
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

scatter = ax.scatter(encoded_images_pca[:, 0], encoded_images_pca[:, 1], c=kmeans_labels, cmap='viridis')

# Add widgets for interactive clustering
axcolor = 'lightgoldenrodyellow'
ax_clusters = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
slider_clusters = widgets.Slider(ax_clusters, 'Clusters', 2, 20, valinit=10, valstep=1)

def update(val):
    n_clusters = int(slider_clusters.val)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(encoded_images_pca)
    scatter.set_array(kmeans_labels)
    fig.canvas.draw_idle()

slider_clusters.on_changed(update)

plt.show()
