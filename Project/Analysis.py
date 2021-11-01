

#----------------------------------------  MODULES  ----------------------------------------


#Basic modules:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping

#My functions:
from AMS_functions import NN_output_to_AMS, plot_AMS
from Plot_distributions import plot_distributions, plot_distributions_final
from Splitting_function import splitting
from Splitting_jets_function import splitting_jets, preprocessing_function
from Make_model import make_model

pd.options.display.max_columns = None


#----------------------------------------  DATA PREPARATION  ----------------------------------------


#Let's have a look at the dataset:
data_full = pd.read_csv('dataset_higgs_challenge.csv')

#For this classification I used only yhe "t" (training data), "b" (validation data) and "v" (test data) set of variables:
print('Total number of events: ', len(data_full), '\n')
for KaggleSetID in ['t', 'b', 'v', 'u']:
    print('Number of events in the {} KaggleSet: {}'
          .format(KaggleSetID, len(data_full['KaggleSet'][data_full['KaggleSet']==KaggleSetID])))
          
#Description of the sub-dataset in each line:
#1) Splitting of the dataset into train, test and validation set.
#2) Extracting the weights of the validation and test set.
#3) Extracting the binary arrays for my networks.
#4) Extracting the binary arrays for my BDT
#Within the splitting of the dataset, have been applyied some operations on the engineering of the features for each subset. The problem is that the "phi" variables have a signal distribution that is very similar to the background one. So it's better to consider their linear combination (difference in this case) to make them useful in my classification.
X, df_empty, y_train, y_train_BDT = splitting (data_full, "t")
X_val, weights_val, y_val, y_val_BDT = splitting (data_full, "b")
X_test, weights_test, y_test, y_test_BDT = splitting (data_full, "v")
del(data_full)


#----------------------------------------  BDT  ----------------------------------------


#Let's first scale my data:
standard = StandardScaler()
standard.fit(X)
X_standard = standard.transform(X)
X_val_standard = standard.transform(X_val)
X_test_standard = standard.transform(X_test)

#BDT classification:
BDT = HistGradientBoostingClassifier(max_iter=90, verbose=1, l2_regularization=0.5, learning_rate=.1, max_leaf_nodes=50, random_state=45, max_depth=15, 					      max_bins=50)
BDT.fit(X_standard, y_train_BDT)

y_pred_val = BDT.predict_proba(X_val_standard)
y_pred_test = BDT.predict_proba(X_test_standard)

del X_standard, X_val_standard, X_test_standard

#I will split the results just to be able to combine them with the DNN result later:
BDT_0jets_val = y_pred_val[ X_val['PRI_jet_num']==0 ]
BDT_1jet_val = y_pred_val[ X_val['PRI_jet_num']==1 ]
BDT_2jets_val = y_pred_val[ X_val['PRI_jet_num']>=2 ]

y_pred_BDT_val = np.concatenate((BDT_0jets_val, BDT_1jet_val, BDT_2jets_val))

BDT_0jets_test = y_pred_test[ X_test['PRI_jet_num']==0 ]
BDT_1jet_test = y_pred_test[ X_test['PRI_jet_num']==1 ]
BDT_2jets_test = y_pred_test[ X_test['PRI_jet_num']>=2 ]

y_pred_BDT_test = np.concatenate((BDT_0jets_test, BDT_1jet_test, BDT_2jets_test))


#----------------------------------------  DATA PROCESSING  ----------------------------------------


#Let's construct the data for the case with 0 jets:
X_0jets, y_train_0jets, empty_0 = splitting_jets(X, y_train, df_empty, 0)
X_val_0jets, y_val_0jets, weights_0jets_val = splitting_jets(X_val, y_val, weights_val, 0)
X_test_0jets, y_test_0jets, weights_0jets_test = splitting_jets(X_test, y_test, weights_test, 0)

#Let's construct the data for the case with 1 jets:
X_1jet, y_train_1jet, empty_1 = splitting_jets(X, y_train, df_empty, 1)
X_val_1jet, y_val_1jet, weights_1jet_val = splitting_jets(X_val, y_val, weights_val, 1)
X_test_1jet, y_test_1jet, weights_1jet_test = splitting_jets(X_test, y_test, weights_test, 1)

#Let's construct the data for the case with 2 jets:
X_2jets, y_train_2jets, empty_2 = splitting_jets(X, y_train, df_empty, 2)
X_val_2jets, y_val_2jets, weights_2jets_val = splitting_jets(X_val, y_val, weights_val, 2)
X_test_2jets, y_test_2jets, weights_2jets_test = splitting_jets(X_test, y_test, weights_test, 2)

del empty_0, empty_1, empty_2


#----------------------------------------  2-JETS DNN  ----------------------------------------


#Scaling data:
standard_2jets = StandardScaler()
standard_2jets.fit(X_2jets)
X_2jets_standard = standard_2jets.transform(X_2jets)
X_val_2jets_standard = standard_2jets.transform(X_val_2jets)
X_test_2jets_standard = standard_2jets.transform(X_test_2jets)

#DNN:
np.random.seed(42)
DNN_2jets = make_model([64, 128, 64, 64, 32, 8], 'relu', 0.1, 'Adam', 'L2', 0.0001, X_2jets.shape[-1])

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

history = DNN_2jets.fit(X_2jets_standard, y_train_2jets, batch_size=256, epochs=50, verbose=1, 
                  validation_data=(X_val_2jets_standard, y_val_2jets), callbacks = [early_stopping], class_weight = None)

y_pred_2jets_val = DNN_2jets.predict(X_val_2jets_standard)
y_pred_2jets_test = DNN_2jets.predict(X_test_2jets_standard)

del X_2jets_standard, X_val_2jets_standard, X_2jets, X_val_2jets, X_test_2jets_standard, X_test_2jets


#----------------------------------------  1-JET DNN  ----------------------------------------


#Scaling data:
standard_1jet = StandardScaler()
standard_1jet.fit(X_1jet)
X_1jet_standard = standard_1jet.transform(X_1jet)
X_val_1jet_standard = standard_1jet.transform(X_val_1jet)
X_test_1jet_standard = standard_1jet.transform(X_test_1jet)

#DNN:
np.random.seed(42)
DNN_1jet = make_model([64, 64, 64, 32, 8], 'relu', 0.1, 'Adagrad', 'L1', 0.0001, X_1jet.shape[-1])

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

history = DNN_1jet.fit(X_1jet_standard, y_train_1jet, batch_size=256, epochs=50, verbose=1, 
                  validation_data=(X_val_1jet_standard, y_val_1jet), callbacks = [early_stopping], class_weight = None)

y_pred_1jet_val = DNN_1jet.predict(X_val_1jet_standard)
y_pred_1jet_test = DNN_1jet.predict(X_test_1jet_standard)

del X_1jet_standard, X_val_1jet_standard, X_1jet, X_val_1jet,  X_test_1jet_standard, X_test_1jet


#----------------------------------------  0-JET DNN  ----------------------------------------


#Scaling data:
standard_0jets = StandardScaler()
standard_0jets.fit(X_0jets)
X_0jets_standard = standard_0jets.transform(X_0jets)
X_val_0jets_standard = standard_0jets.transform(X_val_0jets)
X_test_0jets_standard = standard_0jets.transform(X_test_0jets)

#DNN:
np.random.seed(42)
DNN_0jets = make_model([32, 64, 128, 64, 32, 8], 'elu', 0.1, 'Adagrad', 'L1', 0.0001, X_0jets.shape[-1])

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

history = DNN_0jets.fit(X_0jets_standard, y_train_0jets, batch_size=256, epochs=50, verbose=1, 
                  validation_data=(X_val_0jets_standard, y_val_0jets), callbacks = [early_stopping], class_weight = None)

y_pred_0jets_val = DNN_0jets.predict(X_val_0jets_standard)
y_pred_0jets_test = DNN_0jets.predict(X_test_0jets_standard)

del X_0jets_standard, X_val_0jets_standard, X_0jets, X_val_0jets, X_test_0jets_standard, X_test_0jets


#----------------------------------------  TOTAL AMS SCORE OF DNNS  ----------------------------------------


#Total AMS score considering all the AMS of each subset:
y_pred_DNN_val = np.concatenate((y_pred_0jets_val, y_pred_1jet_val, y_pred_2jets_val))
y_val_total = np.concatenate((y_val_0jets, y_val_1jet, y_val_2jets))
weights_total_val = np.concatenate((weights_0jets_val, weights_1jet_val, weights_2jets_val))

y_pred_DNN_test = np.concatenate((y_pred_0jets_test, y_pred_1jet_test, y_pred_2jets_test))
y_test_total = np.concatenate((y_test_0jets, y_test_1jet, y_test_2jets))
weights_total_test = np.concatenate((weights_0jets_test, weights_1jet_test, weights_2jets_test))


#----------------------------------------  COMBINING DNNs AND BDT AMS  ----------------------------------------


dataset_blend_val = np.append(y_pred_DNN_val[:,1].reshape(-1,1), y_pred_BDT_val[:,1].reshape(-1,1), axis=1)
dataset_blend_test = np.append(y_pred_DNN_test[:,1].reshape(-1,1), y_pred_BDT_test[:,1].reshape(-1,1), axis=1)
blend = LogisticRegression(solver='lbfgs')
blend.fit(dataset_blend_val,  y_val_total[:,1])
blended_val = blend.predict_proba(dataset_blend_val)
blended_test = blend.predict_proba(dataset_blend_test)


#----------------------------------------  FINAL RESULTS  ----------------------------------------


print('DNN:')
plot_AMS(y_pred_DNN_test, y_test_total, weights_total_test)
print('BDT:')
plot_AMS(y_pred_BDT_test, y_test_total, weights_total_test)
print('Combination:')
plot_AMS(blended_test, y_test_total, weights_total_test)
plt.legend(['DNN', 'BDT', 'DNN + BDT'])
plt.ylim(2.8,)
plt.savefig('AMS_total.png', dpi=300)
plt.show()

plot_distributions_final(blended_val, blended_test, y_val_total, 50, False, weights_total_val, 
                         weights_total_test)
plt.savefig('Final_distribution_unweighted.png', dpi=300)
plt.show()

plot_distributions_final(blended_val, blended_test, y_val_total, 50, True, weights_total_val, 
                         weights_total_test)
plt.savefig('Final_distribution_weighted.png', dpi=300)
plt.show()
