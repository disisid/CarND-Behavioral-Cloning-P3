import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Model
import matplotlib.pyplot as plt

lines = []
with open('data_given/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
train_samples, validation_samples = train_test_split(lines, test_size=0.20009955)
images = []
measurements = []
for line in lines:
  center_image = line[0]
  left_image = line[1]
  right_image = line[2]
  #filename = source_path.split('/')[-1]
  current_path = 'data_given/IMG/'
  img_center = cv2.imread(current_path + center_image.split('/')[-1] )
  img_left = cv2.imread(current_path + left_image.split('/')[-1])
  img_right = cv2.imread(current_path + right_image.split('/')[-1])
  images.extend((img_center,img_left,img_right))
  measurement =float(line[3])
  correction = 0.3
  measurement_l = measurement+correction
  measurement_r = measurement-correction
  measurements.extend((measurement,measurement_l,measurement_r))
  #augment dataset with lr flipped images
  img_center_f = np.fliplr(img_center)
  img_left_f = np.fliplr(img_left)
  img_right_f = np.fliplr(img_right) 
  images.extend((img_center_f,img_left_f,img_right_f))
  measurement_flipped = -measurement
  measurement_flipped_l =-measurement_l
  measurement_flipped_r=-measurement_r
  measurements.extend((measurement_flipped,measurement_flipped_l,measurement_flipped_r))


def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # name = 'data_given/IMG/'+batch_sample[0].split('/')[-1]
                # center_image = cv2.imread(name)
                # center_angle = float(batch_sample[3])
                # images.append(center_image)
                # measurements.append(center_angle)
                center_image = batch_sample[0]
                left_image = batch_sample[1]
                right_image = batch_sample[2]
                #filename = source_path.split('/')[-1]
                current_path = 'data_given/IMG/'
                img_center = cv2.imread(current_path + center_image.split('/')[-1])
                img_left = cv2.imread(current_path + left_image.split('/')[-1])
                img_right = cv2.imread(current_path + right_image.split('/')[-1])
                images.append(img_center)
                images.append(img_left)
                images.append(img_right)
                measurement =float(line[3])
                correction = 0.3
                measurement_l = measurement+correction
                measurement_r = measurement-correction
                measurements.append(measurement)
                measurements.append(measurement_l)
                measurements.append(measurement_r)
                #augment dataset with lr flipped images
                #img = cv2.flip(img, 1)

                # img_center_f = cv2.flip(img_center, 1)
                # img_left_f = cv2.flip(img_left, 1)
                # img_right_f = cv2.flip(img_right, 1)
                # images.append(img_center_f)
                # images.append(img_left_f)
                # images.append(img_right_f)
                # measurement_flipped = -measurement
                # measurement_flipped_l =-measurement_l
                # measurement_flipped_r=-measurement_r
                # measurements.append(measurement_flipped)
                # measurements.append(measurement_flipped_l)
                # measurements.append(measurement_flipped_r)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)



X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
#implementing nvidia based CNN model
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Convolution2D(36,5,5,activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.compile(loss='mse',optimizer='adam')
checkpoint = ModelCheckpoint('model_new{epoch:02d}.h5')
model.fit(X_train, y_train,validation_split=0.2,shuffle=True,nb_epoch=3,callbacks=[checkpoint])

model.save('model_new_nvidia.h5')

# history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=5, verbose=1)

# ### print the keys contained in the history object
# print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
