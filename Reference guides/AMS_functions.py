import numpy as np
from matplotlib import pyplot as plt

def NN_output_to_AMS(x_cut, predictions, label_vectors, weights):
    
    b_reg = 10
    
    s = sum(weights[(predictions[:,1] > x_cut)  & (label_vectors[:,1] ==1)])
    b = sum(weights[(predictions[:,1] > x_cut)  & (label_vectors[:,1] ==0)])

    AMS = np.sqrt(  2 *( (s + b + b_reg) * np.log(1 + s/(b + b_reg)) -s )  )
    
    return AMS


def plot_AMS(predictions, label_vectors, weights):
    x = np.arange(0.5,1,1e-2)
    y = np.array(  [ NN_output_to_AMS(x_values, predictions, label_vectors, weights) for x_values in x ]  )
    
    plt.plot(x, y)
    plt.xlabel('Cut Parameter')
    plt.ylabel('AMS Score')
    plt.grid()
    
    print('The best AMS Score is {:.3f} at a Cut Parameter of {:.2f}'.format(max(y), x[np.argmax(y)]))
