# Utils

## Table of contents

- [AMS_functions.py](#ams-functionspy)
- [Plot_distributions.py](#plot-distributionspy)
- [Make_model.py](#make-modelpy)
- [Splitting_function.py](#splitting-functionpy)
- [Splitting_jets_function.py](#splitting-jets-functionpy)
- [Function testing](#function-testing)

## [AMS_functions.py](https://github.com/JustWhit3/higgs-decay-classification/blob/master/utils/AMS_functions.py)

This document contains the definition and explanation of the AMS functions.
AMS metric is used for the evaluation of my model. To see its detailed definition see the [PDF_dataset.pdf](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc/PDF_dataset.pdf) document.

In this document have been defined two functions:

- `NN_output_to_AMS`, this function takes 4 arguments:

  - "x_cut" is the cut parameter of the AMS. It ranges from 0.5 to 1 in steps of 0.1.
  - "predictions" is a binary array, defined from the set of data that we're considering (for ex: validation set).
  - "label_vectors" is a binary array constructed from the dataset, used for each model, that distinguishes an event between signal and background.
  - "weights" this takes the weights associated to each data of my dataset (in my case the "KaggleWeight").

- `plot_AMS`: this function takes similar arguments of the previous one. It uses the previous one to plot the final result of the AMS.

## [Plot_distributions.py](https://github.com/JustWhit3/higgs-decay-classification/blob/master/utils/Plot_distributions.py)

This document contains the definition and explanation of the "plot_distribution" functions.
There are two functions defined in my program: `plot_distributions` and `plot_distributions_final`.This functions are useful for the plotting of the distributions of each model. I'll explain only the second one, because it's more complete and extended in respect to the first one. So, this one takes 7 arguments:

- "prediction_val" that are prediction data for the validation set. It's a 2-dim array.
- "prediction_test" that are prediction data for the test set. It's a 2-dim array.
- "true_val" that are the output data of the model (for the validation set). It's a 2-dim array.
- "n_bins" that are the number of bins (usually set to 50). It's an integer.
- "weighted" is a boolean variable set to be True if the histogram is weighted, otherwise if it's unweighted.
- "weights_val" in case in which my histogram is weighted this are the weights of the validation data.
- "weights_test" and this are the weights of the test data.

## [Make_model.py](https://github.com/JustWhit3/higgs-decay-classification/blob/master/utils/Make_model.py)

This document contains the definition and explanation of the "make_model" functions. It has been defined this function, in a way to avoid to repeat, every time you define a new DNN model, the same code. This function takes 7 arguments:

- "layer_sizes" is related to the size of the layers of the network. It takes a list of integers.
- "activation" is related to the activation function that you use. It takes a string with the name of the activation function.
- "dropout rate" this is the rate of the dropout, if 0, there will be no dropout.
- "optimizer" this takes a string with the name of the optimizer you want to use.
- "regularization" this takes a string with the name of the regularizer you want to use.
- "input_dimension" this takes the shape of the input data.

## [Splitting_function.py](https://github.com/JustWhit3/higgs-decay-classification/blob/master/utils/Splitting_function.py)

This document contains the definition and explanation of the "splitting" function. This function is useful for the splitting of the dataset into a subset (train, test or validation set). In particular, given a certain dataset, this gives:

- The selected subset that you want.
- The binary vectors both for the networks and the BDT.
- The weights associated to the validation and test set.

In this function it has been applied also an operation on the feature engineering: the problem is that the "phi" variables, in the Kaggle dataset, have a signal distribution that is very similar to the background one. So it's better to consider their linear combination (difference in this case) to make them useful in my classification:

- `Delta_phi_tau_lep` = `PRI_tau_phi` - `PRI_lep_phi` (not helpful for 2 jets category)
- `Delta_phi_met_lep` = `PRI_met_phi` - `PRI_lep_phi`
- `Delta_phi_jet_jet` = `PRI_jet_leading_phi` - `PRI_jet_subleading_phi`

Drop `PRI_tau_phi`,  `PRI_lep_phi`, `PRI_met_phi`, `PRI_jet_leading_phi` and `PRI_jet_subleading_phi`

It depends on 2 variables:

- "dataset": it's the name of the dataset. It receives a DataFrame from Pandas.
- "string": it's related to the the kind of subset you want. It's a string, you've to put the letter of the test, training or validation set in my case..

## [Splitting_jets_function.py](https://github.com/JustWhit3/higgs-decay-classification/blob/master/utils/Splitting_jets_function.py)

This document contains the definition of two functions:

- `preprocessing_function`: it's used for the preprocessing of the dataset, used before applying all the more significant cuts on its features (like the dropping of some of them). This function sets the -999 value of `MMC` variable to the mean improved classification and drops the useless features from the subset. It takes only one argument, that is the previous mentioned subset (pandas DataFrame).
- The function for the splitting into jets (for the DNNs only): `splitting_jets`. This function is useful to split my dataset, keeping into account the number of jets, in the final state of the Higgs boson decay. The dataset is divided into three sets of data: one for the 0 jets case, one for the 1 jet case and one for the 2 or 3 jets case. This function takes 4 arguments:

  - "subset": is one of the set obtained from the previous splitting of the dataset with the `splitting` function. It's a pandas DataFrame.
  - "y_subset": is the binary array for the subset.
  - "weights": is the weight of the validation or test set.
  - "jets_number": indicates the number of jets. It's an integer.

## Function testing

For the function testing has been used the [`pytest`](https://github.com/pytest-dev/pytest) tool.

If you want to test the functionality of the functions you've to do this passages:

- Download the pytest tool from your shell, typing: `pip install pytest`.
- Write on the shell: `pytest <function>.py` where <function<function>> represents the name of the function you want to test and ".py" is the file extension.
