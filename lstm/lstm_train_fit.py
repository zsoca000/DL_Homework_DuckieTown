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
data_codes = np.load(dir+'names.npy') # the ids of runs

#=========================DATA_READING_AND_TRAINING============================


b_s = 1 # the actula size of batch

x_train = []
y_train = []

n = 1 # the number of loaded datas

# iterating trough all datas
for n in data_codes:
   i = 1
   name_x = ""
   name_y = ""
   while (path.isfile(name_x) and path.isfile(name_y)) or i==1:
     
     name_x = dir + 'x/' + str(n) + '_' + str(i) + '.jpg'  # images
     name_y = dir + 'y/' + str(n) + '_' + str(i) + '.npy'  # keyboard commands
     
     if path.isfile(name_x) and path.isfile(name_y):
       
       if b_s == batch_size + 1:
          
          # convert the datas to be acceptable for the lstm model
          tmp_x = []
          for j in range(len(x_train)-5):
              tmp_x.append([x_train[j],x_train[j+1],x_train[j+2],x_train[j+3],x_train[j+4]])

          x_train = tmp_x
          y_train = y_train[5:]
          
          x_train = np.array(x_train).reshape(len(x_train),time_steps,40,80,3)
          y_train = keras.utils.to_categorical(y_train,5)
          
          model.train_on_batch(x_train, y_train) # train the model
          
          # refresh variables
          x_train = []
          y_train = []
          b_s = 1

       # load datas
       img = Image.open(name_x)
       x = img_to_array(img)
       x /= 255
       x_train.append(x)

       y = np.load(name_y)
       y= np.argmax(y)
       y_train.append(y)
       b_s += 1
         
          
       print('load( '+str(n)+' ): ' + name_x, end='\r')
       n+=1  
       i+=1

     
model.save(dir + 'reinf_learning_model') # save the model
