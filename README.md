<p align="center"><img src="https://github.com/JustWhit3/higgs-decay-classification/blob/master/img/logo.svg" height=220></p>

<h3 align="center">Classification of Higgs boson decays using machine learning. Project for the "Tandem Project" activity at Master degree in Physics.</h3>
<p align="center">
    <img title="v2.0" alt="v2.0" src="https://img.shields.io/badge/version-v2.0-informational?style=flat-square"
    <a href="LICENSE">
        <img title="MIT License" alt="license" src="https://img.shields.io/badge/license-MIT-informational?style=flat-square">
    </a>
	<img title="Python 3.8" alt="Python 3.8" src="https://img.shields.io/badge/Python-3.8-informational?style=flat-square">
    </a></br>
	<img title="Code size" alt="code size" src="https://img.shields.io/github/languages/code-size/JustWhit3/higgs-decay-classification?color=red">
	<img title="Repo size" alt="repo size" src="https://img.shields.io/github/repo-size/JustWhit3/higgs-decay-classification?color=red">
	<img title="Lines of code" alt="total lines" src="https://img.shields.io/tokei/lines/github/JustWhit3/higgs-decay-classification?color=red">
</p>

## Table of contents

- [Introduction](#introduction)
- [Repository diagram structure](#repository-diagram-structure)
- [Documentation](#documentation)
- [Unofficial paper](#unofficial-paper)
- [Credits](#credits)
  - [Project leaders](#project-leaders)
  - [Other contributors](#other-contributors)

## Introduction

Study of the Higgs boson Yukawa coupling to tau leptons using the 2012 ATLAS Run-2 dataset. Particular focus is dedicated to the usage of machine learning classification algorithms to classify the Higgs decay channel H to tautau as signal with respect to the other background processes.

For the classification have been considered the cases in which there are 0,1 or 2 jets in the final state.

This classification has been performed on the free dataset from the [Higgs Boson Challenge](https://www.kaggle.com/c/higgs-boson/overview) ([dataset](http://opendata.cern.ch/record/328)), that contains data related to the case in which we have in the final state a tau that decays hadronically and the other one that decays leptonically (data for same leptonic or hadronic decays of the tau are omitted).

Analysis scripts are located into the [python](https://github.com/JustWhit3/higgs-decay-classification/blob/master/scripts/python) folder, while Jupyter Notebooks examples are located into the [jupyter](https://github.com/JustWhit3/higgs-decay-classification/blob/master/scripts/jupyter) folder. The purpose of this latter is to show interactively the various analysis passages.

The software is and will stay **free**, but if you want to support me with a donation it would be really appreciated!

<a href="https://www.buymeacoffee.com/JustWhit33" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

## Repository diagram structure

```
higgs-decay-classification/
├── doc/
│   ├── PDF_dataset.pdf
│   ├── background_explanation.md
│   ├── run_the_code.md
│   ├── utils.md
│   ├── CREDITS.md
│   ├── CONTRIBUTING.md
├── img/
├── scripts/
│   ├── jupyter/
│   │   ├── analysis.ipynb
│   │   ├── plots.ipynb
│   ├── python/
│   │   ├── analysis.py
│   │   ├── plots.py
├── utils/
│   ├── AMS_functions.py
│   ├── Make_model.py
│   ├── Plot_distributions.py
│   ├── Splitting_function.py
│   ├── Splitting_jets_function.py
│── README.md
│── LICENSE
│── CITATION.cff
│── .gitignore
│── .gitattributes
```
 
## Documentation

List of documentation from the [doc](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc) folder:

- [Background explanation](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc/background_explanation.md): contains a detailed background explanation of the analysis.
- [How to run the code](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc/run_the_code.md): contains information about how to run the code on your device.
- [Utils explanation](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc/utils.md): contains information about the functions defined for the main program.
- [Pdf of the challenge](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc/dataset.pdf): it is a pdf containing information about the dataset and the challenge.
- [Contributing](https://github.com/JustWhit3/higgs-decay-classification/blob/master/doc/CONTRIBUTING.md): contains information about how to contribute to this project.

## Final results:

Final weighted distribution:

<img src="https://github.com/JustWhit3/higgs-decay-classification/blob/master/img/unweighted.png" height=400>

## Unofficial paper

An unofficial paper has been produced within this analysis. It has been presented at the [2020 ISHEP](https://www.unibo.it/it/didattica/insegnamenti/insegnamento/2020/453478) school through a small presentation.

This paper can be accessed [here](https://www.researchgate.net/publication/344397759_Tandem_Project_Report_Classification_in_particle_physics_using_machine_learning).

## Credits

### Project leaders

<table>
  <tr>
    <td align="center"><a href="https://justwhit3.github.io/"><img src="https://avatars.githubusercontent.com/u/48323961?v=4" width="100px;" alt=""/><br /><sub><b>Gianluca Bianco</b></sub></a></td>
    <td align="center"><a href="https://github.com/FloMau"><img src="https://avatars.githubusercontent.com/u/49448420?v=4" width="100px;" alt=""/><br /><sub><b>FloMau</b></sub></a></td>
  </tr>
</table>

### Other contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://mlliarm.github.io/"><img src="https://avatars.githubusercontent.com/u/5833270?v=4" width="100px;" alt=""/><br /><sub><b>Mi. Lia.</b></sub></a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
