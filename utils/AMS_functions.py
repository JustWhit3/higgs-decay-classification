#AMS metric is used for the evaluation of my model. To see its definition see the "PDF_dataset.pdf" document into the "Explanations" folder.
#In this document have been defined two functions:

#1) `NN_output_to_AMS`: 
#this function takes 4 arguments:
# 1. "x_cut" is the cut parameter of the AMS. It ranges from 0.5 to 1 in steps of 0.1.
# 2. "predictions" is a binary array, defined from the set of data that we're considering (for ex: validation set).
# 3. "label_vectors" is a binary array constructed from the dataset, used for each model, that distinguishes an event between signal and background.
# 4. "weights" it takes the weights associated to each data of my dataset (in my case the "KaggleWeight").

#2) `plot_AMS`: this function takes similar arguments of the previous one. It uses the previous function to plot the final result of the AMS.

import numpy as np
from matplotlib import pyplot as plt


def NN_output_to_AMS(x_cut, predictions, label_vectors, weights):
    '''Useful to calculate the AMS score.'''
    
    b_reg = 10
    
    s = sum(weights[(predictions[:,1] > x_cut)  & (label_vectors[:,1] ==1)])
    b = sum(weights[(predictions[:,1] > x_cut)  & (label_vectors[:,1] ==0)])

    AMS = np.sqrt(  2 *( (s + b + b_reg) * np.log(1 + s/(b + b_reg)) -s )  )
    
    return AMS


def plot_AMS(predictions, label_vectors, weights):
    '''Useful to plot the AMS function.'''
    x = np.arange(0.5,1,1e-2)
    y = np.array(  [ NN_output_to_AMS(x_values, predictions, label_vectors, weights) for x_values in x ]  )
    
    plt.plot(x, y)
    plt.xlabel('Cut Parameter')
    plt.ylabel('AMS Score')
    plt.grid()
    
    print('The best AMS Score is {:.3f} at a Cut Parameter of {:.2f}'.format(max(y), x[np.argmax(y)]))


#TESTING "NN_OUTPUT_TO_AMS":
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series, range_indexes

@given(
       x=st.floats(0.5, 1),
       y=arrays(np.float64, (1,100000), elements=st.floats(0,1), fill=None, unique=False),
       z=arrays(np.int8, (1,100000), elements=None, fill=None, unique=False), 
       t=series(elements=None, dtype=np.float64, index=range_indexes(min_size=1, max_size=1), fill=None, unique=False)
      )
@settings(deadline=None)
def test_NN_output_to_AMS(x,y,z,t):
    b_reg = 10
    s = sum(t[(y[:,1] > x)  & (z[:,1] ==1)])
    b = sum(t[(y[:,1] > x)  & (z[:,1] ==0)])
    AMS = np.sqrt(  2 *( (s + b + b_reg) * np.log(1 + s/(b + b_reg)) -s )  )
    assert NN_output_to_AMS(x,y,z,t) == AMS




