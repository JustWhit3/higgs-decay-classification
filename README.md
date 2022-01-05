# higgs-decay-classification

## Table of contents

- [Introduction](#introduction)
- [Repository diagram structure](#repository-diagram-structure)
- [Documentation](#documentation)
- [Results](#results)

## Introduction

Study of the Higgs boson Yukawa coupling to tau leptons using the 2012 ATLAS Run-2 dataset. Particular focus is dedicated to the usage of machine learning classification algorithms to classify the Higgs decay channel H to tautau as signal with respect to the other background processes.

For the classification have been considered the cases in which there are 0,1 or 2 jets in the final state.

This classification has been performed on the free dataset from the [Higgs Boson Challenge](https://www.kaggle.com/c/higgs-boson/overview) ([dataset](http://opendata.cern.ch/record/328)), that contains data related to the case in which we have in the final state a tau that decays hadronically and the other one that decays leptonically (data for same leptonic or hadronic decays of the tau are omitted).

## Repository diagram structure

```
higgs-decay-classification/
├── doc/
│   ├── PDF_dataset.pdf
│   ├── background_explanation.md
│   ├── run_the_code.md
├── img/
│   ├── accuracy_0jets.png
│   ├── accuracy_1jet.png
│   ├── accuracy_2jet.png
│   ├── distributions_variables.png
│   ├── s_c_final_AMS.png
│   ├── s_c_unweighted.png
│   ├── s_c_weighted.png
├── scripts/
│   ├── Analysis.ipynb
│   ├── Analysis.py
├── utils/
│   ├── AMS_functions.py
│   ├── Make_model.py
│   ├── Plot_distributions.py
│   ├── Splitting_function.py
│   ├── Splitting_jets_function.py
│── README.md
│── LICENSE
│── .gitignore
│── .gitattributes
├── setup.ls
```
 
## Documentation

List of documentation from the [doc](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc) folder:

- [Background explanation](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc/background_explanation.md): contains a detailed background explanation of the analysis.
- [How to run the code](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc/run_the_code.md): contains information about how to run the code on your device.
- [Utils explanation](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc/utils.md): contains information about the functions defined for the main program.
- [Pdf of the challenge](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc/dataset.pdf): it is a pdf containing information about the dataset and the challenge.

## Results

Final results plots:

Unweighted distribution for signal-background discrimination for validation set:

![alt text](https://github.com/JustWhit3/higgs-decay-classification/blob/master/img/s_c_unweighted.png)

Weighted distribution for signal-background discrimination for validation set:

![alt text](https://github.com/JustWhit3/higgs-decay-classification/blob/master/img/s_c_weighted.png)
