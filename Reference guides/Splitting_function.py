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

    return 0

#TESTING "SPLITTING" FUNCTION:
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series, range_indexes, columns, data_frames

@given(
       x=data_frames(columns=columns(["EventId", "DER_mass_MMC", "DER_mass_transverse_met_lep",
       'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt',
       'Weight', 'Label', 'KaggleSet', 'KaggleWeight'],
       dtype=float),
       rows=st.tuples(st.floats(allow_nan=False), st.floats(allow_nan=False)).map(sorted)),
       y=st.text(),
       z=st.text(), 
      )
def test_splitting(x,y,z):
    assert splitting(x,y,z) == 0
