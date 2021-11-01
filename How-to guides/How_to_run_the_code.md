## Steps to run the code on your device
### 0) Download the dataset
You first need to download the CERN Open Dataset from the [site](http://opendata.cern.ch/record/328).

Than move the dataset into your Home folder, and rename it as: "dataset_higgs_challenge.csv".
### 1) Download the file
You need to download the file from the repository of GitHub.

Click on the button "Clone or download" and than click on "Download ZIP".
### 2) Extract the program from the download file
You need a software like "Winrar" to extract the files from the ZIP document.

Once you've done this, you need to search for the file "Program.ipynb", into the "Project" folder.
### 3) Run the code on your computer
To run the code you need first to download the Jupyter Notebook from [here](https://jupyter.org/install). You can use Conda or Pip.

Than you've to tip this command on the bash:
```shell
jupyter-notebook
```

It's better if you put your executable code in the Home of your computer, in a way to make easy to find it. Be sure that the libraries with the definition of the functions like "AMS_functions.py", "Make_model.py", "Plot_distributions.py" and others are in the same folder of the "Program.ipynb".

Once the page of Jupyter has been opened, you can select the "Program.ipynb" and click with the left botton on it.

Alternatively you can run the .py source code. In this case you have first to set up the Python environment path:
```shell
source setup.ls
```
and then to run the code:
```shell
python3 Analysis.py
```
