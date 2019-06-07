import os
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple as nt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Data = nt("Data", "x_train y_train x_valid y_valid x_test y_test")

def visualize_data(data):
    images_to_show = 36
    per_row = 12
    fig = plt.figure(figsize=(20,5))
    for i in range(images_to_show):
        pos = (i // per_row, ((i % per_row) + per_row) % per_row)
        ax = plt.subplot2grid((int(images_to_show / per_row), per_row),
                              pos, xticks=[], yticks=[])
        ax.imshow(np.squeeze(data.x_train[i]))
    plt.show()

# A chart showing how the accuracy for the training and tests sets evolved
def visualize_training(hist):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.show()
    
    # A chart showing our training vs validation loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

from keras.layers import BatchNormalization
from keras.optimizers import rmsprop
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


def optimized_preprocess(data, categories):
    # Z-score normalization of data
    mean = np.mean(data.x_train, axis=(0,1,2,3))
    std = np.std(data.x_train, axis=(0,1,2,3))
    x_train = ((data.x_train - mean) / (std + 1e-7)).astype("float32")
    x_test = ((data.x_test - mean) / (std + 1e-7)).astype("float32")
    y_train = to_categorical(data.y_train, categories)
    y_test = to_categorical(data.y_test, categories)    
    return Data(x_train[5000:], y_train[5000:],
                x_train[:5000], y_train[:5000],
                x_test, y_test)

def learningrate_schedule(epoch):
    # We use a standard learning rate of 0.001
    # From the 51st epoch, we decrease it to 0.0007
    # From the 101st epoch, we decrease it further to 0.0005
    # From the 1356h epoch, we decrease it further to 0.0003
    # From the 176th epoch, we decrease it further to 0.0001
    rate = 0.001
    if epoch > 175:
        rate = 0.0001
    elif epoch > 135:
        rate = 0.0003
    elif epoch > 100:
        rate = 0.0005
    elif epoch > 50:
        rate = 0.0007    
    return rate

def build_optimized_cnn(data, categories):
    # Create model architecture
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay), input_shape=data.x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, padding="same", activation="elu",
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(categories, activation="softmax"))

    # Compile the model, using an optimized rms this time, which we will adapt
    # during training
    optimized_rmsprop = rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss="categorical_crossentropy", optimizer=optimized_rmsprop,
                  metrics=["accuracy"])
    return model


# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train = x_train[:25000], y_train[:25000]
print(np.unique(y_train, return_counts=True))
data = Data(x_train, y_train, None, None, x_test, y_test)

# Visualize the data
# visualize_data(data)

# Preprocess the data
categories = len(np.unique(data.y_train))
print("Shape of x_train pre-processing: ", data.x_train.shape)
print("Shape of y_train pre-processing: ", data.y_train.shape)
# Preprocess for optimized cnn
optimized_processed_data = optimized_preprocess(data, categories)
print("Shape of x_train post-processing: ", optimized_processed_data.x_train.shape)
print("Shape of y_train post-processing: ", optimized_processed_data.y_train.shape)
print("Shape of x_valid post-processing: ", optimized_processed_data.x_valid.shape)
print("Shape of y_valid post-processing: ", optimized_processed_data.y_valid.shape)
print("Shape of x_test post-processing: ", optimized_processed_data.x_test.shape)
print("Shape of y_test post-processing: ", optimized_processed_data.y_test.shape)

# Build optimized cnn
optimized_cnn = build_optimized_cnn(optimized_processed_data, categories)
print("Optimized CNN architecture:")
optimized_cnn.summary()

# Perform data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.15,
                             height_shift_range=0.15, horizontal_flip=True)
datagen.fit(optimized_processed_data.x_train)


cnn_weights_path = "saved_weights/cifar10_cnn_best.hdf5"

# Train the optimized cnn
batch_size = 64
optimized_cnn_path_best = "saved_weights/optimized_cifar10_cnn_best.hdf5"
checkpointer_optimized_cnn = ModelCheckpoint(optimized_cnn_path_best, verbose=1, 
                                             save_best_only=True)
hist_optimized_cnn = optimized_cnn.fit_generator(datagen.flow(optimized_processed_data.x_train, 
    optimized_processed_data.y_train, batch_size=batch_size), 
    steps_per_epoch=optimized_processed_data.x_train.shape[0] // batch_size, epochs=250,
    verbose=1, validation_data= (optimized_processed_data.x_valid,
    optimized_processed_data.y_valid), callbacks=[checkpointer_optimized_cnn,
    LearningRateScheduler(learningrate_schedule), EarlyStopping(min_delta=0.001, 
    patience=40)])

visualize_training(hist_optimized_cnn)

optimized_cnn.load_weights(optimized_cnn_path_best)
score_optimized_cnn = optimized_cnn.evaluate(optimized_processed_data.x_test, 
                                             optimized_processed_data.y_test, verbose=0)

print("Accuracy optimized cnn: {0:.2f}%".format(score_optimized_cnn[1] * 100))