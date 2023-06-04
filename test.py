import numpy as np
from sklearn.metrics import f1_score

# Load the saved model
model = tf.keras.models.load_model('animal_classification_model.h5')

# Make predictions on the test set
test_images, test_labels = [], []
for images, labels in test_generator:
    test_images.append(images)
    test_labels.append(labels)
test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)
predictions = model.predict(test_images)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Convert one-hot encoded labels to class labels
true_classes = np.argmax(test_labels, axis=1)

# Calculate the F1 score
f1 = f1_score(true_classes, predicted_classes, average='weighted')

# Print the F1 score
print(f"F1 Score: {f1}")
