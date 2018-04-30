
# coding: utf-8

# # Hand-sign Detection
# # *---------------------------*

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'
nb_train_samples = 2000
nb_validation_samples = 400
nb_test_samples = 440
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# # Data Preprocessing

# In[2]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# # Building Model

# In[3]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# # Model Summary

# In[4]:


model.summary()


# # Training model

# In[5]:


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# In[6]:


model.save('first_try.h5')


# # Model Accuracy on Test dataset

# In[7]:


prediction = model.predict_generator(test_generator)


# # Testing single image

# In[8]:


from keras.models import load_model
import keras.backend as K
from keras.preprocessing import image 
import numpy as np
import matplotlib.pyplot as plt

loaded_model = load_model("first_try.h5")


# # 'One sign' Image

# In[9]:


import numpy as np
from keras.preprocessing import image


test_image = image.load_img('Test-Images/one/one7.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)

plt.imshow(test_image.reshape(150, 150,3).astype('uint8'))

print("Available classes" ,train_generator.class_indices)
print("image belongs to class ",result)
if result[0][0] == 1:
    prediction = 'peace'
else:
    prediction = 'one'
print ("Sign is ",prediction)


# # 'Peace sign' Image

# In[10]:


import numpy as np
from keras.preprocessing import image
test_image2 = image.load_img('Test-Images/peace/peace10.jpg', target_size = (150, 150))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)
result2 = model.predict(test_image2)

plt.imshow(test_image2.reshape(150, 150,3).astype('uint8'))

print("Available classes" ,train_generator.class_indices)
print("image belongs to class ",result2)
if result2[0][0] == 1:
    prediction2 = 'peace'
else:
    prediction2 = 'one'
print ("Sign is ",prediction2)

