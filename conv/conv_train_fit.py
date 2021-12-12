
#===============================DEPNDENCIES==================================

from  conv_model import conv_model
import numpy as np
from os import path
from PIL import Image
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping

#===============================INITIALIZING=================================

epochs = 100
batch_size = 512
model = conv_model()

dir = 'datas/'
data_codes = np.load(dir+'names.npy')

#===============================DATA_READING=================================

x_train = []
y_train = []

for n in data_codes:

   i = 1
   name_x = dir + 'x/' + str(n) + '_' + str(i) + '.jpg'
   name_y = dir + 'y/' + str(n) + '_' + str(i) + '.npy' 
   
   while path.isfile(name_x) and path.isfile(name_y):
     
     img = Image.open(name_x)
     x = img_to_array(img)
     x /= 255
     x_train.append(x)

     y = np.load(name_y)
     y= np.argmax(y)
     y_train.append(y) 
     
     i+=1
     name_x = dir + 'x/' + str(n) + '_' + str(i) + '.jpg'
     name_y = dir + 'y/' + str(n) + '_' + str(i) + '.npy'
     
     print(len(x_train),'--->',name_x)

print(len(x_train))

#================================TRAINING====================================

y_train = keras.utils.to_categorical(y_train,5)
x_train = np.array(x_train).reshape(len(x_train),40,80,3)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=8)
model.fit(x_train,
          y_train,
          epochs=epochs,
          validation_split = 0.2,
          batch_size=batch_size,  
          callbacks = early_stopping,        
          verbose=1,
          shuffle=True)
          
model.save(dir + 'reinf_learning_model')


