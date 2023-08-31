import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout, Reshape, Concatenate, \
    LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Model


# Height and width refer to the size of the image
# Channels refers to the amount of color channels (red, green, blue)

# image_dimensions = {'height':256, 'width':256, 'channels':3}

# Create a Classifier class

class Classifier:
    def __init__(self=None):
        self.model = 0

    def predict(self, x):
        #         print(x.shape)
        return self.model.predict(x)

    def fit(self, x, y):
        #         print(x.shape)
        #         print(y.shape)
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)


# Create a MesoNet class using the Classifier

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def init_model(self):
        mod = Sequential()

        mod.add(Conv2D(filters=8, kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu'))
        mod.add(BatchNormalization())
        mod.add(MaxPool2D((2, 2), padding='same'))

        mod.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu'))
        mod.add(BatchNormalization())
        mod.add(MaxPool2D((2, 2), padding='same'))

        mod.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'))
        mod.add(BatchNormalization())
        mod.add(MaxPool2D((2, 2), padding='same'))

        mod.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'))
        mod.add(BatchNormalization())
        mod.add(MaxPool2D((4, 4), padding='same'))

        mod.add(Flatten())

        mod.add(Dropout(0.5))
        mod.add(Dense(16))
        mod.add(LeakyReLU(alpha=0.1))
        mod.add(Dropout(0.5))
        mod.add(Dense(1, activation='sigmoid'))

        return mod


# Instantiate a MesoNet model with pretrained weights
meso = Meso4()
# meso.load('../input/deepfake/Deepfake-detection/weights/Meso4_DF')
# Prepare image data

# Rescaling pixel values (between 1 and 255) to a range between 0 and 1
dataGenerator = ImageDataGenerator(rescale=1. / 255)

# Instantiating generator to feed images through the network
generator = dataGenerator.flow_from_directory(
    '../input/deepfake/Deepfake-detection/data/',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary')
# Found 7104 images belonging to 3 classes.
# Checking class assignment
generator.class_indices
{'.ipynb_checkpoints': 0, 'DeepFake': 1, 'Real': 2}

# Recreating generator after removing '.ipynb_checkpoints'
dataGenerator = ImageDataGenerator(rescale=1. / 255)

generator = dataGenerator.flow_from_directory(
    '../input/deepfake/Deepfake-detection/data/',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary')

# Re-checking class assignment after removing it
generator.class_indices

# Rendering image X with label y for MesoNet
X, y = generator.next()

# Evaluating prediction
print(f"Predicted likelihood: {meso.predict(X)[0][0]:.4f}")
print(f"Actual label: {int(y[0])}")
print(f"\nCorrect prediction: {round(meso.predict(X)[0][0]) == y[0]}")

# # Showing image
plt.imshow(np.squeeze(X));
