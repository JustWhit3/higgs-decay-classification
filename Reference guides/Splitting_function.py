#This function is useful for the splitting of the dataset into a subset (train, test or validation set). In
#particular, given a certain dataset, this returns:
#1) The selected subset that you want. It's a Pandas DataFrame.
#2) The weights associated to the validation and test set. It's a Pandas DataFrame.
#3-4) The binary vectors both for the networks and the BDT.

#In this function it has been applyied also an operation on the feature enginnering:
#The problem is that the "phi" variables, in the Kaggle dataset, have a signal distribution that is very similar 
#to the background one. So it's better to consider their linear combination (difference in this case) to make 
#them useful in my classification.

#It depends on 2 variables:
#1) "dataset": it's the name of the dataset. It receives a DataFrame from Pandas.
#2) "string": it's related to the the kind of subset you want. It's a string, you've to put the letter of the subset
#you want.

import pandas as pd
import numpy as np

def splitting (dataset, string):
    '''Splits dataset into subsets.'''

    subset = dataset[dataset['KaggleSet']==string]
    
    #I have to construct binary arrays for my network.
    y_subset = pd.get_dummies(subset['Label']).values.astype(float) #signal = [0,1], background = [1,0]
    
    ##### for BDT #### It needs 1D classes and can handle strings, so I can use the 's' or 'b' labels.
    y_subset_BDT = subset['Label']
    
    X = subset.iloc[:,1:-4]
    
    if string == "b" or string == "v":
        weights = subset['KaggleWeight']
        Event_IDs = subset['EventId']
    elif string == "t":
        weights = pd.DataFrame({'A' : [], 'B' : []})
        
    #Feature enginnering.
    X['Delta_phi_tau_lep'] = abs(X['PRI_tau_phi']-X['PRI_lep_phi'])
    X['Delta_phi_jet_jet'] = abs(X['PRI_jet_leading_phi']-X['PRI_jet_subleading_phi'])
    X['Delta_phi_met_lep'] = abs(X['PRI_met_phi']-X['PRI_lep_phi'])
    X['Delta_eta_tau_lep'] = abs(X['PRI_tau_eta']-X['PRI_lep_eta'])
    
    #Fnal conditions for the output.
    return X, weights, y_subset, y_subset_BDT
        
    del subset


#TESTING "SPLITTING" FUNCTION.
def test_splitting():
    df = pd.DataFrame({'KaggleSet': ['t','b','v'], 'Label' : ['s', 'b', 's'], 'KaggleWeight' : [0.1, 0.2, 0.3], 'EventId' : [1, 2, 3],
                       'PRI_tau_phi': [1, 2, 3], 'PRI_lep_phi': [1, 2, 3], 'PRI_jet_leading_phi' : [1, 2, 3], 'PRI_jet_subleading_phi' : [1, 2, 3],
                       'PRI_met_phi' : [1, 2, 3], 'PRI_tau_eta' : [1, 2, 3], 'PRI_lep_eta' : [1, 2, 3], 'empty_1' : [1, 2, 3],
                       'empty_2' : [1, 2, 3], 'empty_3' : [1, 2, 3], 'empty_4' : [1, 2, 3]})

    #Let's check the case of training set.
    res_1 = pd.DataFrame({'Label' : ['s'], 'KaggleWeight' : [0.1], 'EventId' : [1], 'PRI_tau_phi': [1], 'PRI_lep_phi': [1], 'PRI_jet_leading_phi' : [1], 'PRI_jet_subleading_phi' : [1],
                          'PRI_met_phi' : [1], 'PRI_tau_eta' : [1], 'PRI_lep_eta' : [1], 'Delta_phi_tau_lep' : [0], 'Delta_phi_jet_jet' : [0], 'Delta_phi_met_lep' : [0],
                          'Delta_eta_tau_lep' : [0]})
    res_2 = pd.DataFrame({'A' : [], 'B' : []})
    res_3 = np.array([[1.]])
    res_4 = df[df['KaggleSet']=='t']['Label']

    expected_res = res_1, res_2, res_3, res_4
    pd.testing.assert_series_equal(pd.Series(splitting(df, 't')), pd.Series(expected_res), check_names=False)
  
    #Let's check the case of validation set.
    res_1_val = pd.DataFrame({'Label' : ['b'], 'KaggleWeight' : [0.2], 'EventId' : [2], 'PRI_tau_phi': [2], 'PRI_lep_phi': [2], 'PRI_jet_leading_phi' : [2], 'PRI_jet_subleading_phi' : [2],
                              'PRI_met_phi' : [2], 'PRI_tau_eta' : [2], 'PRI_lep_eta' : [2], 'Delta_phi_tau_lep' : [0], 'Delta_phi_jet_jet' : [0], 'Delta_phi_met_lep' : [0],
                              'Delta_eta_tau_lep' : [0]}, index=pd.Index(range(1,2)))
    res_2_val = df[df['KaggleSet']=='b']['KaggleWeight']
    res_3_val = res_3
    res_4_val = df[df['KaggleSet']=='b']['Label']

    expected_res_val = res_1_val, res_2_val, res_3_val, res_4_val
    pd.testing.assert_series_equal(pd.Series(splitting(df, 'b')), pd.Series(expected_res_val), check_names=False)

    #Let's check the case of test set.
    res_1_test = pd.DataFrame({'Label' : ['s'], 'KaggleWeight' : [0.3], 'EventId' : [3], 'PRI_tau_phi': [3], 'PRI_lep_phi': [3], 'PRI_jet_leading_phi' : [3], 'PRI_jet_subleading_phi' : [3],
                              'PRI_met_phi' : [3], 'PRI_tau_eta' : [3], 'PRI_lep_eta' : [3], 'Delta_phi_tau_lep' : [0], 'Delta_phi_jet_jet' : [0], 'Delta_phi_met_lep' : [0],
                              'Delta_eta_tau_lep' : [0]}, index=pd.Index(range(2,3)))
    res_2_test = df[df['KaggleSet']=='v']['KaggleWeight']
    res_3_test = res_3
    res_4_test = df[df['KaggleSet']=='v']['Label']
    res_4_test.index = [2]

    expected_res_test = res_1_test, res_2_test, res_3_test, res_4_test
    pd.testing.assert_series_equal(pd.Series(splitting(df, 'v')), pd.Series(expected_res_test), check_names=False)
