
#===============================DEPNDENCIES==================================
import os
from  conv_model import conv_model
import numpy as np
from os import path
from PIL import Image
from tensorflow import keras
from keras.preprocessing.image import img_to_array
#from keras.callbacks import EarlyStopping

#===============================INITIALIZING=================================

epochs = 60
batch_size = 512
patience = 15
val_split = 5 # menet
model = conv_model()

dir = 'datas/'
data_codes = np.load(dir+'names.npy')
pat_count = 0
best_loss = 10e10
best_acc = 0
os.system('clear')
#===============================DATA_READING=================================
for epoch in range(epochs):
  print("Epoch:"+str(epoch+1)+"/"+str(epochs))
  name = dir + 'names'
  run_num = 175 #len(data_codes)
  randname = str(data_codes[run_num])

#================================TRAINING====================================
  count = 1
  losses = []
  accs = []
  while (run_num <= len(data_codes)-1):	
    x_train = []
    y_train = []
    found=0
    
    for i in range(batch_size):
      	

      namex = dir + 'x/' + randname + '_' + str(count) + '.jpg'
      namey = dir + 'y/' + randname + '_' + str(count) + '.npy'
      
      if (path.isfile(namex) and path.isfile(namey)):
        
        img = Image.open(namex)
        x = img_to_array(img)
        x /= 255
        x_train.append(x)
        found+=1
        
        y = np.load(namey)
        y= np.argmax(y)
        y_train.append(y)        

        count += 1
      else:
        run_num += 1
        
        if run_num > len(data_codes)-1: break
        randname = str(data_codes[run_num])
        count = 1
        namex = dir + 'x/' + randname + '_' + str(count) + '.jpg'
        namey = dir + 'y/' + randname + '_' + str(count) + '.npy'
        
        if (path.isfile(namex) and path.isfile(namey)):
        	img = Image.open(namex)
        	x = img_to_array(img)
        	x /= 255
        	x_train.append(x)
        	found+=1
        
        
        	y = np.load(namey)
        	y= np.argmax(y)
        	y_train.append(y)
        
        	count += 1

    y_train = keras.utils.to_categorical(y_train,5)
    x_train = np.array(x_train).reshape(found,40,80,3)
    if run_num < len(data_codes)-val_split:  # train

      model.train_on_batch(x_train, y_train)
      print("trained runs:   "+str(run_num),end="\r")
    elif (run_num < len(data_codes)-1): # validation
      loss,acc = model.evaluate(x_train, y_train,batch_size=batch_size)
      losses.append(loss)
      accs.append(acc)
      
   
  losses=np.array(losses)
  accs=np.array(accs)
  loss = losses.mean()
  acc = accs.mean()
  print("validated       loss=", str(loss), "   accuracy=", str(acc), "   patience=", str(pat_count), end="\r")
  if acc > best_acc:
     best_acc = acc
     pat_count = 0
     print("\n----------------------------------saving-----------------------------\n")
     model.save(dir + 'reinf_learning_model')
     print("\n----------------------------------saved------------------------------\n")
  else:
     pat_count += 1
   
  if(pat_count >= patience):
     print("EARLY STOPPING      loss:", str(best_loss), "   accuracy=", str(acc), end="\n")
     break
      
# save model

