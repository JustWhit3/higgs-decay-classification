#It has been defined this function, in a way to avoid to repeat everytime you define a new DNN model the same code. This function takes 7 arguments:

#+ "layer_sizes" is related to the size of the layers of the network. It takes a list of integers.
#+ "activation" is related to the activation function that you use. It takes a string with the name of the activation function.
#+ "dropout rate" this is the rate of the dropout, if 0, there will be no dropout.
#+ "optimizer" this takes a string with the name of the optimizer you want to use.
#+ "regularization" this takes a string with the name of the regularizer you want to use.
#+ "input_dimension" this takes the shape of the input data.

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

#TESTING "MAKE_MODEL" (DOENS'T WORK):
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series, range_indexes

@given (
	x=st.lists(elements=st.integers(8,128), min_size=1, max_size=6),
	y=st.just('relu'),
	z=st.just(0.1),
	t=st.just('Adam'),
	#k=st.functions(like=lambda x: "0.0001", returns=st.just(0.1)),
	k=st.text(),
	w=st.just(0.0001),
	n=st.integers(1,100)
       )
def test_make_models(x,y,z,t,k,w,n):
	model_ = Sequential()
	if k == 'L1': regularizer_ = l1(w)
	elif k == 'L2': regularizer_ = l2(w)

	model_.add(Dense(units=x[0], activation=y, kernel_regularizer=regularizer_ , input_dim=n))

	for layer_size in x[1:]:
		model_.add(Dense(layer_size, activation=y, kernel_regularizer=regularizer_ ))
		model_.add(Dropout(z))
		model_.add(Dense(2))
		model_.add(Activation('softmax'))

	model_.compile(loss='binary_crossentropy', optimizer=t, metrics=['accuracy'])

	assert make_model(x,y,z,t,k,w,n) == model_
