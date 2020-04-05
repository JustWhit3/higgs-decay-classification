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

#TESTING "SPLITTING_JETS" FUNCTION (DOESN'T WORK):
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series, range_indexes, columns, data_frames, indexes

@given(
       x=data_frames(columns=columns(['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt',
       'Delta_phi_tau_lep', 'Delta_phi_jet_jet', 'Delta_phi_met_lep',
       'Delta_eta_tau_lep'],
       dtype=float),
       rows=st.tuples(st.floats(allow_nan=False), st.floats(allow_nan=False)).map(sorted)),
       y=data_frames(columns=columns(['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt',
       'Delta_phi_tau_lep', 'Delta_phi_jet_jet', 'Delta_phi_met_lep',
       'Delta_eta_tau_lep'],
       dtype=float),
       rows=st.tuples(st.floats(allow_nan=False), st.floats(allow_nan=False)).map(sorted)),
       z=data_frames(columns=columns(['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt',
       'Delta_phi_tau_lep', 'Delta_phi_jet_jet', 'Delta_phi_met_lep',
       'Delta_eta_tau_lep'],
       dtype=float),
       rows=st.tuples(st.floats(allow_nan=False), st.floats(allow_nan=False)).map(sorted)),
       t=st.integers(0,2),
       k=arrays(np.int8, (1,250000), elements=st.floats(1,1), fill=None, unique=False), 
       l=arrays(np.int8, (1,100000), elements=st.floats(1,1), fill=None, unique=False),
       m=arrays(np.int8, (1,450000), elements=st.floats(1,1), fill=None, unique=False),
       a=series(elements=None, dtype=np.float64, index=indexes(elements=None, dtype=np.int64, min_size=1, max_size=1, unique=True), fill=None, unique=False),
       b=series(elements=None, dtype=np.float64, index=indexes(elements=None, dtype=np.int64, min_size=1, max_size=1, unique=True), fill=None, unique=False),
       c=st.text(),
       x_1=data_frames(columns=columns(['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DERU_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt',
       'PRI_tau_eta', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'Delta_phi_tau_lep', 'Delta_phi_met_lep',
       'Delta_eta_tau_lep'],
       dtype=float),
       rows=st.tuples(st.floats(allow_nan=False), st.floats(allow_nan=False)).map(sorted)),
       x_2=data_frames(columns=columns(['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_lep_pt',
       'PRI_lep_eta', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet',
       'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_all_pt', 'Delta_phi_tau_lep',
       'Delta_phi_jet_jet', 'Delta_phi_met_lep', 'Delta_eta_tau_lep'],
       dtype=float),
       rows=st.tuples(st.floats(allow_nan=False), st.floats(allow_nan=False)).map(sorted)),
       y_1=arrays(np.int8, (1,138925), elements=st.floats(1,1), fill=None, unique=False),
       w_1=series(elements=None, dtype=np.float64, index=indexes(elements=None, dtype=np.int64, min_size=1, max_size=1, unique=True), fill=None, unique=False),
      )
@settings(suppress_health_check=(HealthCheck.too_slow,HealthCheck.data_too_large,),deadline=None)
def test_splitting_jets(x,y,z,t,k,l,m,a,b,c,x_1,x_2,y_1,w_1):
    assert (
	    splitting_jets(x,y,z,t,k,l,m,a,b,c) == x_1 or
	    splitting_jets(x,y,z,t,k,l,m,a,b,c) == x_2 or
	    splitting_jets(x,y,z,t,k,l,m,a,b,c) == y_1 or
	    splitting_jets(x,y,z,t,k,l,m,a,b,c) == w_1
	   )
