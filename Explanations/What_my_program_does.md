# Higgs Boson Machine Learning Classification
## General Introduction
The purpose of this program is to perform a classification of the decay of the Higgs boson into 2 tau in respect to the possible background processes that could happen. For the classification have been considered the cases in which there are 0,1 or 2 jets in the final state.
This classification has been performed on the free dataset from the Higgs Boson Challenge: ([Link to the challenge](https://www.kaggle.com/c/higgs-boson/overview) and [Link to the CERN Open Dataset](http://opendata.cern.ch/record/328)), that contains data related to the case in which we have in the final state a tau that decays hadronically and the other one that decays leptonically (data for same leptonic or hadronic decays of the tau are omitted).

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
To split the Dataset, according to the number of jets, into subsets (useful only for DNNs):

• Events with 0 jets (100.000 events).

• Events with 1 jet (78.000 events).

• Events ≥ 2 jets (72.000 events).

To drop features that are meaningless for the new subsets.

• Drop 13 variables for 0 jets.

• Drop 8 variables for 1 jet.

• Keep all the variables for the ≥ 2 jets Set.

First, perform a classification on the whole dataset (without considering subsets) using the Gradient Boosted Decision Trees (BDT).
Than, train 3 Deep Neural Networks, one for each subset considering the number of jets.

## Feature Work (only for DNNs)
Distributions of some of the angular variables are uniform (this is a problem, because I coudn't use them for the discrimination between signal and background). So the idea is to build new features according to relative angles:

• `Delta_phi_tau_lep`

• `Delta_phi_met_lep`

• `Delta_phi_jet_jet`

•` Delta_eta_tau_lep`

They have different distributions for Signal and Background. Finally drop all phi variables: `PRI_tau_phi`, `PRI_lep_phi`, `PRI_jet_leading_phi`, `PRI_jet_subleading_phi`.

Here is shown an example of some distributions of the new variables (here it's possible to see a clear discrimination between signal and background):

![alt text](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/distributions_variables.png)

## Scaling of the Data
Before start working on the DNN and the BDT, the input data have been scaled with a Standard Scaling.

## BDT Model
Has been used the class `HistGradientBoostingClassifier` from the library `sklearn`. This BDT model has been used to train the whole data set. It runs very fast and accurate in respect to the DNN one.

## Neural Network Structure
Have been used 3 DNNs, one for each subset according to the number of jets. Has been used the library `keras`. They are structured as follows:

• Using of 5 and 6 hidden layers.

• Relu and elu activation functions.

• Adam and Adagrad optimizers.

• Dropout and L1 regularization.

• Loss: binary crossentropy.

• Metric: Accuracy.

• 2D softmax output.

• (0, 1) for perfect signal and (1, 0) for perfect background events.

Here are shown the different model accuracy plots for each subset (depending on the number of jets):

Plot for 0 jets classification ([Link](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/accuracy_0jets.png)):

![alt text](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/accuracy_0jets.png)

Plot for 1 jet classification ([Link](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/accuracy_1jet.png)):

![alt text](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/accuracy_1jet.png)

Plot for 2 jets classification ([Link](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/accuracy_2jets.png)):

![alt text](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/accuracy_2jets.png)

## Evaluation of the classification
The combination of the models has been performed using the Logistic Regression on both outpus of DNN and BDT. The class used is `LogisticRegression` from the library `sklearn`.
Evaluation of the classification process is given by a metric called "AMS" (see the 
[PDF_dataset.pdf](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Explanations/PDF_dataset.pdf) or the own explanation in the [Readme](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Reference%20guides/README_reference_guides.md) of the [Reference Guides](https://github.com/JustWhit3/Software_and_Computing_program/tree/master/Reference%20guides)). At the end have been combined all the AMS of each classification procedure with the Logistic Regression method.

## Results
For all the final results, see the [project](https://github.com/JustWhit3/Software_and_Computing_program/tree/master/Project) folder. Numerical results (AMS scores) are:
+ Best AMS Score of the DNN: 3.529 at a Cut Parameter of 0.83.
+ Best AMS Score of the BDT: 3.578 at a Cut Parameter of 0.83.
+ Combination of the two AMS scores with Logistic Regression: 3.652 at a Cut Parameter of 0.88.

Graphical results are shown here:
+ Comparison between total, DNNs and BDT AMS ([Link](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/s_c_final_AMS.png)):

![alt text](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/s_c_final_AMS.png)

+ Unweighted distribution for signal-background discrimination for validation set ([Link](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/s_c_unweighted.png)):

![alt text](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/s_c_unweighted.png)

+ Weighted distribution for signal-background discrimination for validation set ([Link](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/s_c_weighted.png)):

![alt text](https://github.com/JustWhit3/Software_and_Computing_program/blob/master/Project/s_c_weighted.png)

NOTE: in this latter case, it's possibile to see a strange result for the last bin. I've investigated this from my own and I've interpreted it as caused by a bit of overtraning in the last part of the graph.
