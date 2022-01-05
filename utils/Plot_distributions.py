#There are two important plotting functions defined in my program: `plot_distributions` and `plot_distributions_final`. This functions are useful for the plotting of the distributions of each model. 
#I'll show the explanation of only the second one, because it's more complete and extended in respect to the first one. So, this one takes 7 arguments:

#1) "prediction_val" that are prediction data for the validation set. It's a 2-dim array.
#2) "prediction_test" that are prediction data for the test set. It's a 2-dim array.
#3) "true_val" that are the output data of the model (for the validation set). It's a 2-dim array.
#4) "n_bins" that are the number of bins (usually set to 50). It's an integer.
#5) "weighted" is a boolean variable set to be True if the histogram is weighted, otherwise if it's unweighted.
#6) "weights_val" in case in which my histogram is weighted this are the weights of the validation data.
#7) "weights_test" and this are the weights of the test data.

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



