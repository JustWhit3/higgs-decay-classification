# Steps to run the code on your device

## Table of contents

- [Download the dataset](#download-the-dataset)
- [Download the file](#download-the-file)
- [Extract the program from the download file](#extract-the-program-from-the-download-file)
- [Run the code on your computer](#run-the-code-on-your-computer)

## Download the dataset

You first need to download the CERN Open Dataset from the [site](http://opendata.cern.ch/record/328).

Than move the dataset into your Home folder, and rename it as: "dataset_higgs_challenge.csv".

## Download the file

You need to download the file from the repository of GitHub.

Click on the button "Clone or download" and than click on "Download ZIP". Alternatively you can download the latest release by clicking on the right release button in the main page.

## Extract the program from the download file

You need a software like "Winrar" to extract the files from the ZIP document.

Once you've done this, you need to search for the file "Program.ipynb", into the "Project" folder.

## Run the code on your computer

To run the code you need first to download the Jupyter Notebook from [here](https://jupyter.org/install). You can use Conda or Pip.

Than you've to tip this command on the bash:

```shell
jupyter-notebook
```

Once the page of Jupyter has been opened, you can select the "Program.ipynb" and click with the left botton on it.

Alternatively you can run the .py source code. In this case you have first to setup the Python environment path from the main repository directory:

```shell
source setup.ls
```

and then to run the code:

```shell
python3 Analysis.py
```
