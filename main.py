import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Paths to image datasets
bear_dataset_path = 'C:/Users/sandor/Documents/animals/bear'
wolf_dataset_path = 'C:/Users/sandor/Documents/animals/wolf'
fox_dataset_path = 'C:/Users/sandor/Documents/animals/fox'

# Define the parameters
batch_size = 32
epochs = 14
image_height, image_width = 150, 150

# Create data generators with data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/sandor/Documents/animals',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'C:/Users/sandor/Documents/animals',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

log_dir = 'C:/Users/sandor/Documents/animals'
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[tensorboard_callback]  # Add the TensorBoard callback
)
# Save the trained model
model.save('animal_classification_model.h5')

# Load the saved model
model = tf.keras.models.load_model('animal_classification_model.h5')

# Test the model on a sample image
test_image_path = 'C:/Users/sandor/Documents/animals/urs1.jpg'
test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(image_height, image_width))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = tf.expand_dims(test_image, axis=0)
test_image = test_image / 255.0

# Make predictions
predictions = model.predict(test_image)
class_labels = train_generator.class_indices
predicted_class = list(class_labels.keys())[list(class_labels.values()).index(tf.argmax(predictions, axis=1).numpy()[0])]

# Check if one of the three animals is recognized
if predicted_class in ['bear', 'wolf', 'fox']:
    print("Alarm: Animal recognized!")

# Print the predicted class and confidence scores
print(f"Predicted class: {predicted_class}")
print(f"Confidence scores: {predictions[0]}")

# Load the saved model
model = tf.keras.models.load_model('animal_classification_model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_image)

# Print the test loss and accuracy
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

