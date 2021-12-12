from conv_model import conv_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = 'datas/x/'

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)




#test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(40, 80), batch_size=20, class_mode='binary')

#validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(img_height, img_width), batch_size=20, class_mode='binary')

