#Definition of the function for the splitting into jets for the DNNs. This function is useful to split my dataset keeping into account also the number of jets in the final state of the Higgs boson
#decay (this is useful for the DNNs only). The dataset is divided into three sets of data: one for the 0 jets case, one for the 1 jet case and one for the 2 or 3 jets case.
#This function takes several arguments:
#1) "subset": is the training set obtained from the previous splitting of the dataset.
#2) "subset_val": is the validation set obtained from the previous splitting of the dataset.
#3) "subset_test": is the test set obtained from the previous splitting of the dataset.
#3) "jets_number": indicated the number of jets. It's an integer.
#4-6) "y_subset", "y_subset_val" and "y_subset_test": are respectively the binary arrays for the train, validation and test set.
#7-8) "weights_val" and "weights_test": are the weights of the validation and test set.
#9) "thing": it's a string. This takes the name of the object that you want in the output. 

import numpy as np

def splitting_jets (subset, subset_val, subset_test, jets_number, y_subset, y_subset_val, y_subset_test, weights_val, weights_test, thing):
    '''Splits dataset into subsets related to the number of jets.'''
    
    #Setting the -999 value of MMC to the mean improved classification.
    subset['DER_mass_MMC'][subset['DER_mass_MMC']==-999.] = np.mean(subset['DER_mass_MMC'][subset['DER_mass_MMC']!=-999.])
    subset_val['DER_mass_MMC'][subset_val['DER_mass_MMC']==-999.] = np.mean(subset_val['DER_mass_MMC'][subset_val['DER_mass_MMC']!=-999.])
    subset_test['DER_mass_MMC'][subset_test['DER_mass_MMC']==-999.] = np.mean(subset_test['DER_mass_MMC'][subset_test['DER_mass_MMC']!=-999.])
    
    #Drop the unuseful features:
    subset = subset.drop(['PRI_tau_phi', 'PRI_lep_phi', 'PRI_jet_leading_phi', 'PRI_jet_subleading_phi'], axis=1)
    subset_val = subset_val.drop(['PRI_tau_phi', 'PRI_lep_phi', 'PRI_jet_leading_phi', 'PRI_jet_subleading_phi'], axis=1)
    subset_test = subset_test.drop(['PRI_tau_phi', 'PRI_lep_phi', 'PRI_jet_leading_phi', 'PRI_jet_subleading_phi'], axis=1)
    
    #Now I'll drop the unuseful variables with values -999.
    if jets_number == 0:
        ##### 0 jets
        subset_jets = subset[ subset['PRI_jet_num']==0 ].drop(['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 
                                         'DER_lep_eta_centrality', 'PRI_jet_num', 'PRI_jet_leading_pt', 
                                         'PRI_jet_leading_eta', 'PRI_jet_subleading_pt', 
                                         'PRI_jet_subleading_eta', 'PRI_jet_all_pt', 'PRI_jet_num',
                                         'Delta_phi_jet_jet', 'PRI_jet_leading_phi', 'PRI_jet_subleading_phi'], 
                                        axis =1, errors='ignore')
        subset_val_jets = subset_val[ subset_val['PRI_jet_num']==0 ].drop(['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 
                                         'DER_lep_eta_centrality', 'PRI_jet_num', 'PRI_jet_leading_pt', 
                                         'PRI_jet_leading_eta', 'PRI_jet_subleading_pt', 
                                         'PRI_jet_subleading_eta', 'PRI_jet_all_pt', 'PRI_jet_num',
                                         'Delta_phi_jet_jet', 'PRI_jet_leading_phi', 'PRI_jet_subleading_phi'], 
                                        axis =1, errors='ignore')
        subset_test_jets = subset_test[ subset_test['PRI_jet_num']==0 ].drop(['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 
                                         'DER_lep_eta_centrality', 'PRI_jet_num', 'PRI_jet_leading_pt', 
                                         'PRI_jet_leading_eta', 'PRI_jet_subleading_pt', 
                                         'PRI_jet_subleading_eta', 'PRI_jet_all_pt', 'PRI_jet_num',
                                         'Delta_phi_jet_jet', 'PRI_jet_leading_phi', 'PRI_jet_subleading_phi'], 
                                        axis =1, errors='ignore')

        y_train_jets = y_subset[ subset['PRI_jet_num']==0 ]
        y_val_jets = y_subset_val[ subset_val['PRI_jet_num']==0 ]
        y_test_jets = y_subset_test[ subset_test['PRI_jet_num']==0 ]
        weights_jets_val = weights_val[ subset_val['PRI_jet_num']==0 ].reset_index(drop=True)
        weights_jets_test = weights_test[ subset_test['PRI_jet_num']==0 ].reset_index(drop=True)
        
    elif jets_number == 1:
        ##### 1 jet
        subset_jets = subset[ subset['PRI_jet_num']==1 ].drop(['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 
                                        'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 
                                        'DER_lep_eta_centrality', 'PRI_jet_num',
                                        'Delta_phi_jet_jet', 'PRI_jet_subleading_phi'], axis=1, errors='ignore')
        subset_val_jets = subset_val[ subset_val['PRI_jet_num']==1 ].drop(['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 
                                        'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 
                                        'DER_lep_eta_centrality', 'PRI_jet_num',
                                        'Delta_phi_jet_jet', 'PRI_jet_subleading_phi'], axis=1, errors='ignore')
        subset_test_jets = subset_test[ subset_test['PRI_jet_num']==1 ].drop(['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 
                                        'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 
                                        'DER_lep_eta_centrality', 'PRI_jet_num',
                                        'Delta_phi_jet_jet', 'PRI_jet_subleading_phi'], axis=1, errors='ignore')

        y_train_jets = y_subset[ subset['PRI_jet_num']==1 ]
        y_val_jets = y_subset_val[ subset_val['PRI_jet_num']==1 ]
        y_test_jets = y_subset_test[ subset_test['PRI_jet_num']==1 ]
        weights_jets_val = weights_val[ subset_val['PRI_jet_num']==1 ].reset_index(drop=True)
        weights_jets_test = weights_test[ subset_test['PRI_jet_num']==1 ].reset_index(drop=True)
        
        
    elif jets_number == 2:
        ##### 2 and 3 jets
        subset_jets = subset[ subset['PRI_jet_num']>=2 ].drop(['PRI_jet_num',
                                        ], axis=1, errors='ignore')
        subset_val_jets = subset_val[ subset_val['PRI_jet_num']>=2 ].drop(['PRI_jet_num',
                                                     ], axis=1, errors='ignore')
        subset_test_jets = subset_test[ subset_test['PRI_jet_num']>=2 ].drop(['PRI_jet_num',
                                                     ], axis=1, errors='ignore')
        y_train_jets = y_subset[ subset['PRI_jet_num']>=2 ]
        y_val_jets = y_subset_val[ subset_val['PRI_jet_num']>=2 ]
        y_test_jets = y_subset_test[ subset_test['PRI_jet_num']>=2 ]
        weights_jets_val = weights_val[ subset_val['PRI_jet_num']>=2 ].reset_index(drop=True)
        weights_jets_test = weights_test[ subset_test['PRI_jet_num']>=2 ].reset_index(drop=True)
        
    else: None
        
    if thing == "subset": return subset_jets
    elif thing == "subset_val": return subset_val_jets
    elif thing == "subset_test": return subset_test_jets
    elif thing == "binary_train": return y_train_jets
    elif thing == "binary_val": return y_val_jets
    elif thing == "binary_test": return y_test_jets
    elif thing == "weights_val": return weights_jets_val
    elif thing == "weights_test": return weights_jets_test
    else: None

    del subset_jets, subset_val_jets, subset_test_jets
