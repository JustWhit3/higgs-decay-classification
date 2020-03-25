# Higgs Boson Machine Learning Classification
## General Introduction
The purpose of this program is to perform a classification of the decay of the Higgs boson into 2 tau in respect to the possible background processes that could happen. For the classification have been considered the cases in which there are 0,1 or 2 jets in the final state.
This classification has been performed on the free dataset from the Higgs Boson Challenge: https://www.kaggle.com/c/higgs-boson/overview ,that contains data related to the case in which we have in the final state a tau that decays hadronically and the other one that decays leptonically.

## Informations on the dataset
Dataset is divided into some subsets:

• Training Set: KaggleSet = t (used for the training set), 250.000 events.

• Validation Set: KaggleSet = b (used for the validation set), 100.000 events.

• Test Set: KaggleSet = v (used for the test set), 450.000 events.

• Unused: KaggleSet = u (not yet used), 18.000 events.

Informations about the variables:

• 13 Derived and 17 Primitive variables.

• Derived variables are calculated from the primitive variables.

• High correlations.

Signal and Background events are labelled and weighted.

## The Strategy
To split the Dataset according to the number of jets:

• Events with 0 jets (100.000 events).

• Events with 1 jet (78.000 events).

• Events ≥ 2 jets (72.000 events).

To drop features that are meaningless for the new subsets.

• Drop 13 variables for 0 jets.

• Drop 8 variables for 1 jet.

• Keep all the variables for the ≥ 2 jets Set.

First, perform a classification on the whole dataset (without considering subsets) using the Gradient Boosted Decision Trees (BDT).
Than, train 3 Deep Neural Networks, one for each subset considering the number of jets.

## Some Feature Work
Distributions of some of the angular variables are uniform (this is a problem, because I coudn't use them for the discrimination between signal and background). So the idea is to build new features according to relative angles:

• Delta_phi_tau_lep

• Delta_phi_met_lep

• Delta_phi_jet_jet

• Delta_eta_tau_lep

They have different distributions for Signal and Background. Finally drop all phi variables: PRI_tau_phi, PRI_lep_phi, PRI_jet_leading_phi, PRI_jet_subleading_phi.

## Scaling of the Data
Before start working on the DNN and the BDT, the input data have been scaled with a Standard Scaling.

## BDT Model
Has been used the class `HistGradientBoostingClassifier` from the library `sklearn`. This BDT model has been used to train the whole data set. It runs very fast and accurate in respect to the DNN one.

## Neural Network Structure
Have been used 3 DNNs, one for each subset according to the number of jets. Has been used the library `keras`.

• Have been used 5 and 6 hidden layers.

• Relu and elu activation functions.

• Adam and Adagrad optimizers.

• Dropout and L1 regularization.

• Loss: binary crossentropy.

• Metric: Accuracy.

• 2D softmax output.

• (0, 1) for perfect signal and (1, 0) for perfect background events.

## Evaluation of the classification
The combination of the models has been performed using the Logistic Regression on both outpus of DNN and BDT. The class used is `LogisticRegression` from the library `sklearn`.
Evaluation of the classification process is given by a metric called "AMS" (see the "PDF_dataset.pdf" for more indormations or see its definition in the code). At the end have been combined all the AMS of each classification procedure with the Logistic Regression method.