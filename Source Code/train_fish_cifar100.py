import tensorflow as tf
from keras import layers
from keras.applications import VGG16
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2

# Load and preprocess data
(train_X, train_y), (test_X, test_y) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

# Map fine class names to their respective class indices
fine_class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                    'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

# Filter images for the coarse class "fish" and the selected fine classes
selected_fine_classes = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
selected_fine_indices = [fine_class_names.index(cls) for cls in selected_fine_classes]
train_mask = np.isin(train_y.flatten(), selected_fine_indices)
test_mask = np.isin(test_y.flatten(), selected_fine_indices)
train_X, train_y = train_X[train_mask], train_y[train_mask]
test_X, test_y = test_X[test_mask], test_y[test_mask]

# Update label mappings for the selected fine classes
label_mapping = {selected_fine_indices[i]: i for i in range(len(selected_fine_indices))}
train_y = np.vectorize(label_mapping.get)(train_y)
test_y = np.vectorize(label_mapping.get)(test_y)

# Normalize pixel values to [0, 1]
train_X, test_X = train_X / 255.0, test_X / 255.0

# Resize images to match VGG16 input size
train_X_resized = np.array([cv2.resize(img, (224, 224)) for img in train_X])
test_X_resized = np.array([cv2.resize(img, (224, 224)) for img in test_X])

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Build model
model = tf.keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(selected_fine_classes), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Learning Rate Scheduling
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)

# Train model
history = model.fit(train_X_resized, train_y, epochs=20, validation_data=(test_X_resized, test_y), callbacks=[lr_scheduler])

# Evaluate model
test_loss, test_acc = model.evaluate(test_X_resized, test_y)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# Confusion matrix
predictions = np.argmax(model.predict(test_X_resized), axis=1)
cm = confusion_matrix(test_y, predictions)
print(cm)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Visualize predictions
class_names = selected_fine_classes
plt.figure(figsize=(12, 12))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_X_resized[i], cmap=plt.cm.binary)
    true_label = class_names[test_y[i][0]]
    pred_label = class_names[predictions[i]]
    plt.xlabel(f"True: {true_label}\nPred: {pred_label}")
plt.tight_layout()
plt.show()
