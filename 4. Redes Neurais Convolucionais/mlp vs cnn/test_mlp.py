import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple as nt

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
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

def preprocess(data, categories):
    x_train = data.x_train.astype("float32") / 255
    x_test = data.x_test.astype("float32") / 255
    y_train = to_categorical(data.y_train, categories)
    y_test = to_categorical(data.y_test, categories)    
    return Data(x_train[5000:], y_train[5000:],
                x_train[:5000], y_train[:5000],
                x_test, y_test)

def build_mlp(data, categories):
    # Create model architecture
    model = Sequential()
    model.add(Flatten(input_shape=data.x_train.shape[1:]))
    model.add(Dense(320, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(160, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(categories, activation="softmax"))
    
    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", 
                  metrics=["accuracy"])
    return model

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train = x_train[:25000], y_train[:25000]
print(np.unique(y_train, return_counts=True))
data = Data(x_train, y_train, None, None, x_test, y_test)

# Visualize the data
visualize_data(data)

# Preprocess the data
categories = len(np.unique(data.y_train))
print("Shape of x_train pre-processing: ", data.x_train.shape)
print("Shape of y_train pre-processing: ", data.y_train.shape)
processed_data = preprocess(data, categories)
print("Shape of x_train post-processing: ", processed_data.x_train.shape)
print("Shape of y_train post-processing: ", processed_data.y_train.shape)
print("Shape of x_valid post-processing: ", processed_data.x_valid.shape)
print("Shape of y_valid post-processing: ", processed_data.y_valid.shape)
print("Shape of x_test post-processing: ", processed_data.x_test.shape)
print("Shape of y_test post-processing: ", processed_data.y_test.shape)

# Build mlp
mlp = build_mlp(processed_data, categories)
print("MLP architecture:")
mlp.summary()

mlp_weights_path = "saved_weights/cifar10_mlp_best.hdf5"

# Train the mlp
checkpointer_mlp = ModelCheckpoint(filepath=mlp_weights_path, verbose=1, save_best_only=True)
hist_mlp = mlp.fit(processed_data.x_train, processed_data.y_train, batch_size=32, 
                   epochs=20, validation_data=(processed_data.x_valid,
                                                processed_data.y_valid),
                   callbacks=[checkpointer_mlp], shuffle=True)

visualize_training(hist_mlp)

mlp.load_weights(mlp_weights_path)
score_mlp = mlp.evaluate(processed_data.x_test, processed_data.y_test, verbose=0)

print("Accuracy mlp: {0:.2f}%".format(score_mlp[1] * 100))