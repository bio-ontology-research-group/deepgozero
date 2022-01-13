# DeepGOZero: Improving protein function prediction from sequence and zero-shot learning based on ontology axioms

DeepGOZero is a novel method which uses model-theoretic approach for
  learning ontology embeddings and combine it with neural networks for
  protein function prediction. DeepGOZero can exploit formal axioms in
  the GO to make zero-shot predictions, i.e., predict protein
  functions even if not a single protein in the training phase was
  associated with that function.

This repository contains script which were used to build and train the
DeepGOZero model together with the scripts for evaluating the model's
performance.

## Dependencies
* The code was developed and tested using python 3.9.
* To install python dependencies run:
  `pip install -r requirements.txt`
* Install [diamond](https://github.com/bbuchfink/diamond) program on your system (diamond command should be available)


## Data
* https://deepgo.cbrc.kaust.edu.sa/data/deepgozero/ - Here you can find the data
used to train and evaluate our method.
 * data.tar.gz - UniProtKB-SwissProt dataset (release 21_04)
 * data-netgo.tar.gz - NetGO2.0 based dataset

## Scripts
The scripts require GeneOntology in OBO and OWL Formats.
* uni2pandas.py - This script is used to convert data from UniProt
database format to pandas dataframe.
* deepgozero_data.py - This script is used to generate training and
  testing datasets.
* deepgozero.py - This script is used to train the model
* deepgozero_predict.py - This script is used to perform zero-shot predictions
* evaluate.py - The scripts are used to compute Fmax, Smin and AUPR
* evaluate_terms.py - The scripts are used to compute class-centric average AUC
* Normalizer.groovy - The script used to normalize OWL ontology
* Corpus.groovy - This script is used to extract class axiom definitions
* deepgopro.py - This script is used to train the MLP baseline model
* deepgocnn.py - This script is used to train the DeepGOCNN model
* run_diamond.sh - This script is used to obtain Diamond predictions


## Zero-shot predictions
We also make predictions for 2,935 classes without any functional annotations.
Predictions are available for download from here:
* [zero_predictions.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/deepgozero/zero_predictions.tar.gz)


## Citation

If you use DeepGOZero for your research, or incorporate our learning algorithms in your work, please cite:
Maxat Kulmanov, Robert Hoehndorf; DeepGOZero: Improving protein function prediction
  from sequence and zero-shot learning based on ontology axioms
