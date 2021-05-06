import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense

batch_size = 256
num_classes = 3
epochs = 100
input_shape=(x_train.shape[1], 1)

model = Sequential()
intput_shape=(x_train.shape[1], 1)
model.add(Conv1D(128, kernel_size=3,padding = ‘same’,activation=’relu’, input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(128,kernel_size=3,padding = ‘same’, activation=’relu’))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(64, activation=’tanh’))
model.add(Dropout(0.2))
model.add(Dense(32, activation=’tanh’))
model.add(Dropout(0.2))
model.add(Dense(16, activation=’relu’))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation=’softmax’))
model.summary()