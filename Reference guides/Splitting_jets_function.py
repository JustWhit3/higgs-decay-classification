#Definition of the "preprocessing_function": this function is useful to set the -999 value of MMC to the mean improved classification and drop the unuseful features from the subset. It takes only
#one argument, that is the previous mentioned subset (pandas DataFrame).

#Definition of "the splitting_jets" function for the splitting into jets for the DNNs: this function is useful to split my dataset, keeping into account also the number of jets in the final state, of 
#the Higgs boson decay (this is useful for the DNNs only). The dataset is divided into three sets of data: one for the 0 jets case, one for the 1 jet case and one for the 2 or 3 jets case.
#This function takes 4 arguments:
#1) "subset": is one of the set obtained from the previous splitting of the dataset. It's a pandas DataFrame.
#2) "y_subset": is the binary array for the subset.
#3) "weights": is the weight of the validation or test set.
#4) "jets_number": indicated the number of jets. It's an integer.

import numpy as np
import pandas as pd

def preprocessing_function (subset):
    '''Setting the -999 value of MMC to the mean improved classification and drop the unuseful features.'''

    subset['DER_mass_MMC'][subset['DER_mass_MMC']==-999.] = np.mean(subset['DER_mass_MMC'][subset['DER_mass_MMC']!=-999.])
    subset = subset.drop(['PRI_tau_phi', 'PRI_lep_phi', 'PRI_jet_leading_phi', 'PRI_jet_subleading_phi'], axis=1)

    return subset

def splitting_jets (subset, y_subset, weights, jets_number):
    '''Splits dataset into subsets related to the number of jets.'''

    Subset = preprocessing_function(subset)
    
    #Creation of the lists for dropped variables (dropped features are different for 0 or 1 jet).
    list_drop_0jets = ['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 
                                         'DER_lep_eta_centrality', 'PRI_jet_num', 'PRI_jet_leading_pt', 
                                         'PRI_jet_leading_eta', 'PRI_jet_subleading_pt', 
                                         'PRI_jet_subleading_eta', 'PRI_jet_all_pt', 'PRI_jet_num',
                                         'Delta_phi_jet_jet', 'PRI_jet_leading_phi', 'PRI_jet_subleading_phi']

    list_drop_1jet = ['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 
                                        'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 
                                        'DER_lep_eta_centrality', 'PRI_jet_num',
                                        'Delta_phi_jet_jet', 'PRI_jet_subleading_phi']

    #Now I'll drop the unuseful variables with values -999.
    if jets_number == 0:
        subset_jets = Subset[Subset['PRI_jet_num']==0 ].drop(list_drop_0jets, axis =1, errors='ignore')
        y_jets = y_subset[Subset['PRI_jet_num']==0 ]
        weights_jets = weights[Subset['PRI_jet_num']==0 ].reset_index(drop=True)

    elif jets_number == 1:
        subset_jets = Subset[Subset['PRI_jet_num']==1 ].drop(list_drop_1jet, axis=1, errors='ignore')
        y_jets = y_subset[Subset['PRI_jet_num']==1 ]
        weights_jets = weights[Subset['PRI_jet_num']==1 ].reset_index(drop=True)
        
    elif jets_number == 2:
        subset_jets = Subset[Subset['PRI_jet_num']>=2 ].drop(['PRI_jet_num',
                                        ], axis=1, errors='ignore')
        y_jets = y_subset[Subset['PRI_jet_num']>=2 ]
        weights_jets = weights[Subset['PRI_jet_num']>=2 ].reset_index(drop=True)
        
    return subset_jets, y_jets, weights_jets

    del subset_jets


#TESTING "PREPROCESSING_FUNCTION":
def testing_preprocessing_function():
    df = pd.DataFrame({'DER_mass_MMC': [1, 2, 3, 4, -999.], 'PRI_tau_phi': [1, 2, 3, 4, 5], 'PRI_lep_phi': [1, 2, 3, 4, 5],
                   'PRI_jet_leading_phi': [1, 2, 3, 4, 5], 'PRI_jet_subleading_phi': [1, 2, 3, 4, 5]})

    m = np.mean(df['DER_mass_MMC'][df['DER_mass_MMC']!=-999.])
    expected_res = pd.DataFrame({'DER_mass_MMC': [1, 2, 3, 4, m]})
    pd.testing.assert_frame_equal(preprocessing_function(df), expected_res, check_names=False)

#TESTING "SPLITTING_JETS":
def testing_splitting_jets():
    df_x_0 = pd.DataFrame({'Label' : ['s','b','s'], 'KaggleWeight' : [0.1, 0.3, 0.5], 'DER_deltaeta_jet_jet' : [1,2,3], 'DER_mass_jet_jet' : [1,2,3], 'DER_prodeta_jet_jet' : [1,2,3], 
                         'DER_lep_eta_centrality' : [1,2,3],
                         'PRI_jet_leading_pt' : [1,2,3], 'PRI_jet_leading_eta' : [1,2,3], 'PRI_jet_subleading_pt' : [1,2,3], 'PRI_jet_subleading_eta' : [1,2,3], 
                         'PRI_jet_all_pt' : [1,2,3], 'Delta_phi_jet_jet' : [1,2,3], 'PRI_jet_leading_phi' : [1,2,3], 'PRI_jet_subleading_phi' : [1,2,3],
                         'DER_mass_MMC': [1, -999., 3], 'PRI_tau_phi': [1, 2,3], 'PRI_lep_phi': [1, 2,3], 'PRI_jet_num' : [0,1,2]})
    df_y_0 = pd.get_dummies(df_x_0['Label']).values.astype(float)
    df_w_0 = df_x_0['KaggleWeight']
   
    #Expected result for 0 jets.
    res_1 = pd.DataFrame({'Label' : ['s'], 'KaggleWeight' : [0.1], 'DER_mass_MMC': [1.0]})
    res_2 = np.array([[0., 1.]]) #Because I know that [0,1] is for signal and [1,0] for background.
    res_3 = res_1['KaggleWeight']
    expected_res = res_1, res_2, res_3
    pd.testing.assert_series_equal(pd.Series(splitting_jets(df_x_0, df_y_0, df_w_0, 0)), pd.Series(expected_res))

    #Expected result for 1 jet.
    res_1_1jet = pd.DataFrame({'Label' : ['b'], 'KaggleWeight' : [0.3], 'PRI_jet_leading_pt' : [2.0], 'PRI_jet_leading_eta' : [2.0], 'PRI_jet_all_pt' : [2.0], 'DER_mass_MMC': [2.0]},
                              index=pd.Index(range(1,2)))
    res_2_1jet = np.array([[1., 0.]])
    res_3_1jet = res_1_1jet['KaggleWeight']
    res_3_1jet.index = [0]
    expected_res_1jet = res_1_1jet, res_2_1jet, res_3_1jet
    pd.testing.assert_series_equal(pd.Series(splitting_jets(df_x_0, df_y_0, df_w_0, 1)), pd.Series(expected_res_1jet))

    #Expected result for 2 jets.
    res_1_2jet = pd.DataFrame({'Label' : ['s'], 'KaggleWeight' : [0.5], 'DER_deltaeta_jet_jet' : [3], 'DER_mass_jet_jet' : [3], 'DER_prodeta_jet_jet' : [3],'DER_lep_eta_centrality' : [3], 
                               'PRI_jet_leading_pt' : [3],'PRI_jet_leading_eta' : [3],'PRI_jet_subleading_pt' : [3],'PRI_jet_subleading_eta' : [3],'PRI_jet_all_pt' : [3],
                               'Delta_phi_jet_jet' : [3], 'DER_mass_MMC': [3.0]},
                              index=pd.Index(range(2,3)))
    res_2_2jet = res_2
    res_3_2jet = res_1_2jet['KaggleWeight']
    res_3_2jet.index = [0]
    expected_res_2jet = res_1_2jet, res_2_2jet, res_3_2jet
    pd.testing.assert_series_equal(pd.Series(splitting_jets(df_x_0, df_y_0, df_w_0, 2)), pd.Series(expected_res_2jet))
   
