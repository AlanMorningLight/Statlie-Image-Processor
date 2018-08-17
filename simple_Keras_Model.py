from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Conv2D
from keras.utils import to_categorical
from keras.optimizers import rmsprop
from keras.preprocessing.image import ImageDataGenerator

import os

save_dir = os.path.join(os.getcwd(), 'Keras_Trained_model')
model_name = 'keras_cifar10_trained_model.ckpt'

from keras.datasets import cifar10
(x_train, y_train_label), (x_test, y_test_label) = cifar10.load_data()

print("X_train image Shape" , str(x_train.shape) )
print("X_test image Shape" , str(x_test.shape) )

#Perform data normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train_label = to_categorical(y_train_label, 10)
y_test_label = to_categorical(y_test_label, 10)


#Define neural network model

model = Sequential()

model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu' ))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

'''

'''

opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt , metrics=['accuracy'])

model.fit(x_train, y_train_label,
          batch_size=200, nb_epoch=1, verbose=1)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

score = model.evaluate(x_test, y_test_label, verbose=0)
print('CNN Loss value:', score[0])
print('Accuracy:'+ str(score[1]*100) + '%')