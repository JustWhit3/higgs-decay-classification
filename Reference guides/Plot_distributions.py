import numpy as np
from matplotlib import pyplot as plt

def plot_distributions_final(prediction_val, prediction_test, true_val, n_bins, weighted, weights_val, weights_test):
    
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
