# Higgs Boson Machine Learning Classification
## General Introduction
The purpose of this program is to perform a classification of the decay of the Higgs boson into 2 tau in respect to the possible background processes that could happen.
This classification has been performed on the free dataset from the Higgs Boson Challenge: https://www.kaggle.com/c/higgs-boson/overview ,that contains data related to the case in which we have in the final state a tau that decays hadronically and the other one that decays leptonically.

## Informations on the dataset
Dataset is divided into some subsets:

• Training Set: KaggleSet = t (used), 250.000 events.

• Validation Set: KaggleSet = b (used), 100.000 events.

• Test Set: KaggleSet = v (not yet used), 450.000 events.

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

Finally train a Deep Neural Network for each subset.

## Some Feature Work
Distributions of some of the angular variables are uniform (this is a problem, because I coudn't use them for the discrimination between signal and background). So the idea is to build new features according to relative angles:

• Delta_phi_tau_lep

• Delta_phi_met_lep

• Delta_phi_jet_jet

• Delta_eta_tau_lep

They have different distributions for Signal and Background. Finally drop all phi variables: PRI_tau_phi, PRI_lep_phi, PRI_jet_leading_phi, PRI_jet_subleading_phi.

## Scaling of the Data
Before start working on the DNN, the data have been scaled: have been tried all basic scalers, the best working was: Standard Scaling.

## Neural Network Structure
• Have been used 5 and 6 hidden layers.

• Relu and elu activation functions.

• Adam and Adagrad optimizers.

• Dropout and L1 regularization.

• Loss: binary crossentropy.

• Metric: Accuracy.

• 2D softmax output.

• (0, 1) for perfect signal and (1, 0) for perfect background events.

## Evaluation of the classification
Evaluation of the classification process is given by a metric called "AMS" (see the "PDF_dataset.pdf" for more indormations).
