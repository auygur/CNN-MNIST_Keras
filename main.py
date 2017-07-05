import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import mnist
from keras.optimizers import RMSprop

from sklearn import datasets
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline



# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


"""
## Grid Search ##
def create_model1(optimizer='rmsprop'):
    model = Sequential([
        Dense(256, input_shape=(784,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])
 
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
    return model

np.random.seed(5)

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.85, nesterov=True)
model = KerasClassifier(build_fn=create_model1, verbose=0)
optimizers = ['rmsprop', 'adam',sgd]
param_grid = dict(optimizer=optimizers)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
"""

batch_size = 128
num_classes = 10
epochs = 20


# Vanilla Model
model = Sequential([
    Dense(256, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
modelhistory = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: \n', score[0])
print('Test accuracy:', score[1])


#Grid Search for selecting dropout, took a bit long so commented out

"""
def create_model2(optimizer='rmsprop', init='glorot_uniform', dropout=0.2):
    modeldropout = Sequential()
    modeldropout.add(Dense(256, activation='relu', input_shape=(784,)))
    modeldropout.add(Dropout(dropout))

    modeldropout.add(Dense(10, activation='softmax'))
 
    modeldropout.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
                  
    modeldropout.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
                    
    return modeldropout


np.random.seed(5)

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.85, nesterov=True)
model = KerasClassifier(build_fn=create_model2, verbose=0)
optimizers = ['rmsprop', 'adam',sgd]
dropouts = [0.1,0.2,0.3,0.5,0.7]
param_grid = dict(optimizer=optimizers,dropout=dropouts)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
"""

"""
We also used a for loop to determine best number of units in hidden layers.(did for both dropout and non dropout models)

for i in range(16,288,16):
    modeldropout = Sequential()
    modeldropout.add(Dense(i, activation='relu', input_shape=(784,)))
    modeldropout.add(Dropout(0.1))
    modeldropout.add(Dense(10, activation='softmax'))

    modeldropout.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
                  
    modeldropout.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))           
                    
    score = modeldropout.evaluate(x_test, y_test, verbose=0)
    print('Test loss for # of Unit: \n',i, score[0])
    print('Test accuracy for # of Unit:',i, score[1])

"""


#Model with dropout
modeldropout = Sequential()
modeldropout.add(Dense(256, activation='relu', input_shape=(784,)))
modeldropout.add(Dropout(0.1))

modeldropout.add(Dense(10, activation='softmax'))

modeldropout.summary()

modeldropout.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

modeldropouthistory = modeldropout.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = modeldropout.evaluate(x_test, y_test, verbose=0)
print('Test loss: \n', score[0])
print('Test accuracy:', score[1])

df = pd.DataFrame(modelhistory.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")

df = pd.DataFrame(modeldropouthistory.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")



"""
Test loss: 
 0.110773773112
Test accuracy: 0.9784

"""

""" 
Test loss: 
 0.105333590296
Test accuracy: 0.9841

"""
