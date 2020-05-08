import pandas as pd
import cv2
import csv
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
import math
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, MaxPooling2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
import datetime
from random import shuffle

# Load in each entry from the driving log. This contains steering angles and links to the pictures
samples = []
with open('/opt/carnd_p3/data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

# Split off 20% of these to be validation.
# Split happens here so all 3 of left, center and right for each location are either in train or validation
# If they weren't there's a risk of information leak. 
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

start=  datetime.datetime.now()

# This is the factor I used to adjust for the left and right camera. 
correction = 0.2

# Used a generator so that it only loads x pictures at a time. A 32 batch will load 96 images (32 * 3) left, center and right.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                #center images
                name = '/opt/carnd_p3/data/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                
                #left images
                name = '/opt/carnd_p3/data/data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = float(batch_sample[3]) + correction
                
                #right images
                name = '/opt/carnd_p3/data/data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = float(batch_sample[3]) - correction
                
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# Model architecture is taken from the Nvidia end to end self driving research. 

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (160,320,3)))
# The top and bottom of the image are cropped as these do not provide valuable data. 
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5), strides = 2, activation='relu',))
model.add(Conv2D(36,(5,5), strides = 2, activation='relu',))
model.add(Conv2D(48,(5,5), activation='relu',))
model.add(Conv2D(64,(3,3), activation='relu',))
model.add(Conv2D(64,(3,3), activation='relu',))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Fit the model - 3 epochs was sufficient. 
model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=3, verbose=1)

# save the model. 
model.save('model.h5')
end=  datetime.datetime.now()

time= end-start
print(time)