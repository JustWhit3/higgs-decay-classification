## Reference guides folder
### General informations
This folder contains all the informations related to the classes and libraries used for the program.

### List of the documents in the folder
#### AMS_functions.py

This document contains a detailed description of what AMS functions, used in the main code, do.
AMS metric is used for the evaluation of my model. To see its definition see the "PDF_dataset.pdf" document into the "Explanations" folder.
In this document have been defined two functions:

1) `NN_output_to_AMS`: 
this function takes 4 arguments:
+ "x_cut" is the cut parameter of the AMS. It ranges from 0.5 to 1 in steps of 0.1.
+ "predictions" is a binary array, defined from the set of data that we're considering (for ex: validation set).
+ "label_vectors" is a binary array constructed from the dataset, used for each model, that distinguishes an event between signal and background.
+ "weights" this takes the weights associated to each data of my dataset (in my case the "KaggleWeight").

2) `plot_AMS`: this function takes similar arguments of the previous one. It uses the previous one to plot the final result of the AMS.

#### Plot_distributions.py

This document contains a detailed description of what "Plot_distributions" functions, used in the main code, do.
There are two functions defined in my program: `plot_distributions` and `plot_distributions_final`.This functions are useful for the plotting of the distributions of each model. I'll explain only the second one, because it's more complete and extended in respect to the first one. So, this one takes 7 arguments:

+ "prediction_val" that are prediction data for the validation set. It's a 2-dim array.
+ "prediction_test" that are prediction data for the test set. It's a 2-dim array.
+ "true_val" that are the output data of the model (for the validation set). It's a 2-dim array.
+ "n_bins" that are the number of bins (usually set to 50). It's an integer.
+ "weighted" is a boolean variable set to be True if the histogram is weighted, otherwise if it's unweighted.
+ "weights_val" in case in which my histogram is weighted this are the weights of the validation data.
+ "weights_test" and this are the weights of the test data.

#### Make_model.ipynb 

This document contains a detailed description of what" Make_model" function, used in the main code, do. It has been defined this function, in a way to avoid to repeat everytime you define a new DNN model the same code. This function takes 7 arguments:

+ "layer_sizes" is related to the size of the layers of the network. It takes a list of integers.
+ "activation" is related to the activation function that you use. It takes a string with the name of the activation function.
+ "dropout rate" this is the rate of the dropout, if 0, there will be no dropout.
+ "optimizer" this takes a string with the name of the optimizer you want to use.
+ "regularization" this takes a string with the name of the regularizer you want to use.
+ "input_dimension" this takes the shape of the input data.
