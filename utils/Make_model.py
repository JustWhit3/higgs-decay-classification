#It has been defined this function, in a way to avoid to repeat, everytime you define a new DNN model, the same code. This function takes 7 arguments:

#1) "layer_sizes" is related to the size of the layers of the network. It takes a list of integers.
#2) "activation" is related to the activation function that you use. It takes a string with the name of the activation function.
#3) "dropout rate" this is the rate of the dropout, if 0, there will be no dropout.
#4) "optimizer" this takes a string with the name of the optimizer you want to use.
#5) "regularization" this takes a string with the name of the regularizer you want to use.
#6) "input_dimension" this takes the shape of the input data (integer). 

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1, l2
from keras.callbacks.callbacks import EarlyStopping


def make_model(layer_sizes, activation, dropout_rate, optimizer, regularization, lambda_reg, input_dimension):
    '''Creates model comprised of dense layers'''
    model = Sequential()
    if regularization == 'L1':
        regularizer = l1(lambda_reg)
    elif regularization == 'L2':
        regularizer = l2(lambda_reg)
    
    model.add(Dense(units=layer_sizes[0], activation=activation, kernel_regularizer=regularizer , input_dim= input_dimension))
    
    for layer_size in layer_sizes[1:]:
        model.add(Dense(layer_size, activation=activation, kernel_regularizer=regularizer ))
        model.add(Dropout(dropout_rate))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


