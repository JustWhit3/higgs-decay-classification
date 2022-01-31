# Steps to run the code on your device

## Table of contents

- [Download the dataset](#download-the-dataset)
- [Download the file](#download-the-file)
- [Run the code on your computer](#run-the-code-on-your-computer)

## Download the dataset

You first need to download the CERN Open Dataset from the [site](http://opendata.cern.ch/record/328).

Than move the dataset into your Home folder, and rename it as: "dataset_higgs_challenge.csv".

## Download the file

You need to download the file from the repository of GitHub.

Click on the button "Clone or download" and than click on "Download ZIP". Alternatively you can download the latest release by clicking on the right release button in the main page.

You need a software like "Winrar" to extract the files from the ZIP document.

## Run the code on your computer

You have first to to run the analysis code:

```shell
python3 scripts/python/analysis.py
```

With the produced output you can finally run the plots code:

```shell
python3 scripts/python/plots.py
```
