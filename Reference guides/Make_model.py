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
