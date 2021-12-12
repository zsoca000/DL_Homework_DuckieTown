from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Embedding
from tensorflow.keras.optimizers import Adam

def q_model():
  input_shape = (40, 80, 3)
  model = Sequential() 
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, strides=(2, 2))) 

  model.add(Conv2D(64, (3, 3), activation='relu',  strides=(2, 2))) 

  model.add(Conv2D(16, (3, 3), activation='relu')) 
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))

  model.add(Dense(64, activation='relu'))

  model.add(Dense(5, activation='softmax'))

  model.compile(optimizer=Adam(),               
                  loss='mse',
                  metrics=['accuracy'])
  return model,input_shape

#----------------------------------------------------------------------------------------------------

#model = q_model()
# print(model.summary())

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
