#There are two functions defined in my program: `plot_distributions` and `plot_distributions_final`.This functions are useful for the plotting of the distributions of each model. I'll explain only the second one, because it's more complete and extended in respect to the first one. So, this one takes 7 arguments:

#+ "prediction_val" that are prediction data for the validation set. It's a 2-dim array.
#+ "prediction_test" that are prediction data for the test set. It's a 2-dim array.
#+ "true_val" that are the output data of the model (for the validation set). It's a 2-dim array.
#+ "n_bins" that are the number of bins (usually set to 50). It's an integer.
#+ "weighted" is a boolean variable set to be True if the histogram is weighted, otherwise if it's unweighted.
#+ "weights_val" in case in which my histogram is weighted this are the weights of the validation data.
#+ "weights_test" and this are the weights of the test data.

import numpy as np
from matplotlib import pyplot as plt

def plot_distributions_final(prediction_val, prediction_test, true_val, n_bins, weighted, weights_val, weights_test):
    '''Useful to plot the final result, considering also the test set.'''
    # Get histograms from our model
    if weighted:
        hist_b = np.histogram(prediction_val[:,1][true_val[:,0]==1], bins=n_bins, range=(0,1), 
                              weights=weights_val[true_val[:,0]==1])
        hist_s = np.histogram(prediction_val[:,1][true_val[:,1]==1], bins=n_bins, range=(0,1), 
                              weights=weights_val[true_val[:,1]==1])      
        hist_test = np.histogram(prediction_test[:,1], bins=n_bins,range=(0,1), weights=weights_test)
        errorbar_test = np.sqrt(hist_test[0])
    else:
        hist_b = np.histogram(prediction_val[:,1][true_val[:,0]==1], bins=n_bins, range=(0,1))
        hist_s = np.histogram(prediction_val[:,1][true_val[:,1]==1], bins=n_bins, range=(0,1))
        hist_test = np.histogram(prediction_test[:,1], bins=n_bins, range=(0,1))
        errorbar_test = np.sqrt(hist_test[0]/4.5)
        

    bin_edges = hist_b[1]
    bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    # Draw objects
    ax = plt.subplot(111)
    ax.bar(bin_centers, hist_b[0], bottom=hist_s[0], width=bin_widths, alpha=.9)
    ax.bar(bin_centers, hist_s[0], width=bin_widths, alpha=.9)
    if weighted:
        plt.errorbar(bin_centers, hist_test[0], yerr = 5*errorbar_test, fmt = 'k.')
    else:
        plt.errorbar(bin_centers, hist_test[0]/4.5, yerr = 5*errorbar_test, fmt = 'k.')
    plt.xlim(+.04,.96)

    if weighted:
        plt.title("Weighted")
        plt.legend(['Val. Background','Val. Signal',r'Test Set ($5 \times$Errorbars)'])
    else:
        plt.title("Unweighted")
        plt.legend(['Val. Background','Val. Signal',r'Test Set ($5 \times$Errorbars)'])
    plt.xlabel("Model Output")
    plt.ylabel("Counts")
    plt.yscale('log')

    return 0

def plot_distributions(prediction, true, n_bins, weighted, weights):
    '''Useful to plot the final result, without considering also the test set.'''
    
    # Get histograms from our model
    if weighted:
        hist_b = np.histogram(prediction[:,1][true[:,0]==1], bins=n_bins, range=(0,1), weights=weights[true[:,0]==1])
        hist_s = np.histogram(prediction[:,1][true[:,1]==1], bins=n_bins, range=(0,1), weights=50*weights[true[:,1]==1])
        
    else:
        hist_b = np.histogram(prediction[:,1][true[:,0]==1], bins=n_bins, range=(0,1))
        hist_s = np.histogram(prediction[:,1][true[:,1]==1], bins=n_bins, range=(0,1))

    bin_edges = hist_b[1]
    bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    # Draw objects
    ax = plt.subplot(111)
    ax.bar(bin_centers, hist_b[0], width=bin_widths, alpha=.9)
    ax.bar(bin_centers, hist_s[0], bottom=hist_b[0], width=bin_widths, alpha=.9)
    plt.xlim(-.01,1.01)

    if weighted:
        plt.title("Weighted Validation Set Distribution")
        plt.legend(['Background', r'$50\cdot$Signal'])
    else:
        plt.title("Validation Set Distribution (unweighted)")
        plt.legend(['Background', 'Signal'])
    plt.xlabel("DNN Output")
    plt.ylabel("Counts")
    plt.yscale('log')

    return 0

#TESTING "PLOT_DISTRIBUTIONS":
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series, range_indexes

@given(
       x=arrays(np.float64, (1,100000), elements=st.floats(0,1), fill=None, unique=False),
       y=arrays(np.int8, (1,100000), elements=st.floats(1,1), fill=None, unique=False), 
       z=st.integers(1,100),
       t=st.booleans(),
       k=series(elements=st.floats(1,20), dtype=np.float64, index=range_indexes(min_size=1, max_size=1), fill=None, unique=False)
      )
@settings(deadline=None)
def test_plot_distributions(x,y,z,t,k):
    assert plot_distributions(x,y,z,t,k) == 0

#TESTING "PLOT_DISTRIBUTIONS_FINAL":
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series, range_indexes

@given(
       x_val=arrays(np.float64, (1,100000), elements=st.floats(0,1), fill=None, unique=False),
       x_test=arrays(np.float64, (1,100000), elements=st.floats(0,1), fill=None, unique=False),
       y=arrays(np.int8, (1,100000), elements=st.floats(1,1), fill=None, unique=False), 
       z=st.integers(1,100),
       t=st.booleans(),
       k_val=series(elements=st.floats(1,20), dtype=np.float64, index=range_indexes(min_size=1, max_size=1), fill=None, unique=False),
       k_test=series(elements=st.floats(1,20), dtype=np.float64, index=range_indexes(min_size=1, max_size=1), fill=None, unique=False)
      )
@settings(deadline=None)
def test_plot_distributions_final(x_val,x_test,y,z,t,k_val,k_test):
    assert plot_distributions_final(x_val,x_test,y,z,t,k_val,k_test) == 0
