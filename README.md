# DeepGOZero: Improving protein function prediction from sequence and zero-shot learning based on ontology axioms

DeepGOPlus is a novel method for predicting protein functions from
protein sequences using deep neural networks combined with sequence
similarity based predictions.

This repository contains script which were used to build and train the
DeepGOZero model together with the scripts for evaluating the model's
performance.

## Dependencies
* The code was developed and tested using python 3.9.
* To install python dependencies run:
  `pip install -r requirements.txt`
* Install [diamond](https://github.com/bbuchfink/diamond) program on your system (diamond command should be available)


## Data
* http://deepgo.cbrc.kaust.edu.sa/data/deepgozero/ - Here you can find the data
used to train and evaluate our method.
 * data.tar.gz - UniProtKB-SwissProt dataset (release 21_04)
 * data-netgo.tar.gz - NetGO2.0 based dataset

## Scripts
The scripts require GeneOntology in OBO Format.
* uni2pandas.py - This script is used to convert data from UniProt
database format to pandas dataframe.
* deepgozero_data.py - This script is used to generate training and
  testing datasets.
* deepgozero.py - This script is used to train the model
* evaluate.py - The scripts are used to compute Fmax, Smin and AUPR


## Citation

If you use DeepGOZero for your research, or incorporate our learning algorithms in your work, please cite:
Maxat Kulmanov, Robert Hoehndorf; DeepGOZero: Improving protein function prediction
  from sequence and zero-shot learning based on ontology axioms
