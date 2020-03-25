## Steps to solve the problem proposed in the program
This document is useful if you're interested in performing a similar analysis on your dataset. So I'll describe here all the
fundamental passages for the problem solving. The general idea is this one:
### 1) Open the dataset and inspect it
First you need to open the dataset in a proper way, I recommend you to use for example a DataFrame from the library `pandas`.
Pay attention to the variables you're interested for your analysis. Work on them first of all.
### 2) Split the dataset
Split your dataset into different subsets (like test and training set for example).
### 3) Make some feature enginnering
This is useful if you need to transform some variables into others in order to maintain the dependence on the primitive ones
and to improve the classification.
### 4) Construct your models
Construct and perform all the models you want for the classification process (DNNs. BDTs etc...).
### 5) Combine all the outputs
Combine all the results of each model in order to get a better general classification.
### 6) Show the final results
Show the final results with graphs or numbers. Be sure to put error bars and other useful informations on the graphs.
