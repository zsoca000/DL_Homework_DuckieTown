from tensorflow import keras
from keras import Sequential
from keras import Input
from keras.layers import BatchNormalization, ConvLSTM2D, Flatten, Dense, Conv3D

def lstm_model():
  # The hyperparameter space:
  conv_size_1 = 64
  conv_size_2 = 64
  conv_size_3 = 64

  size=(40, 80)
  time_steps = 5
  input = Input(shape=(time_steps, size[0], size[1], 3))

  model = Sequential()
  model.add( ConvLSTM2D(
      filters=conv_size_1,
      kernel_size=(5, 5),
      strides=(2, 2),
      padding="same",
      return_sequences=True,
      activation="relu",
      input_shape=(time_steps, size[0], size[1], 3)
  ))
  model.add(BatchNormalization())
  model.add(ConvLSTM2D(
      filters=conv_size_2,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding="same",
      return_sequences=True,
      activation="relu",
  ))
  model.add(BatchNormalization())
  model.add(ConvLSTM2D(
      filters=conv_size_3,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding="same",
      return_sequences=True,
      activation="relu",
  ))
  model.add(Conv3D(
      filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
  ))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(5, activation='softmax'))

  model.compile(
      loss='mse', optimizer=keras.optimizers.Adam(),
  )

  return model,time_steps
