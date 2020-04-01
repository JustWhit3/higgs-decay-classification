#This function is useful for the splitting of the dataset into a subset (train, test or validation set). In
#particular, given a certain dataset, this gives:
#1) The selected subset that you want.
#2) The binary vectors both for the networks and the BDT.
#3) The weights associated to the validation and test set.
#In this function it has been applyied also an operation on the feature enginnering:
#The problem is that the "phi" variables, in the Kaggle dataset, have a signal distribution that is very similar 
#to the background one. So it's better to consider their linear combination (difference in this case) to make 
#them useful in my classification.

#It depends on 3 variables:
#1) "dataset": it's the name of the dataset. It receives a dataframe from pandas.
#2) "string": it's related from the kind of subset you want. It's a string, you've to put the letter of the subset
#you want.
#3) "thing": this is related to the object that you want the function returns (subset, validation weights,
#binary array for classification etc...). It's a string.

import pandas as pd

def splitting (dataset, string, thing):
    '''Splits dataset into subsets.'''
    subset = dataset[dataset['KaggleSet']==string]
    
    #I have to construct binary arrays for my network.
    y_subset = pd.get_dummies(subset['Label']).values.astype(float) #signal = [0,1], background = [1,0]
    
    ##### for BDT #### It needs 1D classes and can handle strings, so we can use the 's' or 'b' labels
    y_subset_BDT = subset['Label']
    
    X = subset.iloc[:,1:-4]
    
    if string == "b":
        weights_val = subset['KaggleWeight']
        Event_IDs_val = subset['EventId']
    elif string == "v":
        weights_test = subset['KaggleWeight']
        Event_IDs_test = subset['EventId']
    else: 
        None
        
    #Feature enginnering.
    X['Delta_phi_tau_lep'] = abs(X['PRI_tau_phi']-X['PRI_lep_phi'])
    X['Delta_phi_jet_jet'] = abs(X['PRI_jet_leading_phi']-X['PRI_jet_subleading_phi'])
    X['Delta_phi_met_lep'] = abs(X['PRI_met_phi']-X['PRI_lep_phi'])
    X['Delta_eta_tau_lep'] = abs(X['PRI_tau_eta']-X['PRI_lep_eta'])
    
    #Fnal conditions for the output.
    if thing == "subset": return X
    elif thing == "weights validation": return weights_val
    elif thing == "weights test": return weights_test
    elif thing == "binary": return y_subset
    elif thing == "binary BDT": return y_subset_BDT
    else: None
        
    del subset
