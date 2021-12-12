
#===============================DEPNDENCIES==================================

from  lstm_model import lstm_model
import numpy as np
from os import path
from PIL import Image
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping

#===============================INITIALIZING=================================

epochs = 100
batch_size = 128
model, time_steps = lstm_model()

dir = 'datas/'
data_codes = np.load(dir+'names.npy')

#===============================DATA_READING=================================


b_s = 1

x_train = []
y_train = []


for n in data_codes:
   i = 1
   name_x = ""
   name_y = ""
   while (path.isfile(name_x) and path.isfile(name_y)) or i==1:
     
     name_x = dir + 'x/' + str(n) + '_' + str(i) + '.jpg'
     name_y = dir + 'y/' + str(n) + '_' + str(i) + '.npy'
     
     if path.isfile(name_x) and path.isfile(name_y):
       
       if b_s == 256:
          
          tmp_x = []
          for j in range(len(x_train)-5):
              tmp_x.append([x_train[j],x_train[j+1],x_train[j+2],x_train[j+3],x_train[j+4]])

          x_train = tmp_x
          y_train = y_train[5:]
          
          x_train = np.array(x_train).reshape(len(x_train),time_steps,40,80,3)
          y_train = keras.utils.to_categorical(y_train,5)
          
          print('batch learn start')
          model.train_on_batch(x_train, y_train)
          print('batch learn end')
          x_train = []
          y_train = []
          b_s = 1
       
       else:
          
          img = Image.open(name_x)
          x = img_to_array(img)
          x /= 255
          x_train.append(x)

          y = np.load(name_y)
          y= np.argmax(y)
          y_train.append(y)
          b_s += 1
     
     i+=1

     

print(len(x_train))

#=============================TIMESTEP_FORM=====================================

tmp_x = []
for j in range(len(x_train)-5):
  tmp_x.append([x_train[j],x_train[j+1],x_train[j+2],x_train[j+3],x_train[j+4]])

x_train = tmp_x


y_train = y_train[5:]

#================================TRAINING=======================================



x_train = np.array(x_train).reshape(len(x_train),time_steps,40,80,3)
y_train = keras.utils.to_categorical(y_train,5)

"""
early_stopping = EarlyStopping(monitor='val_loss', patience=8)
model.fit(x_train,
          y_train,
          epochs=epochs,
          validation_split = 0.2,
          batch_size=batch_size,  
          callbacks = early_stopping,        
          verbose=1,
          shuffle=True)
"""    
model.save(dir + 'reinf_learning_model')


