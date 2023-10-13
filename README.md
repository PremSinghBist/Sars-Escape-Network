# Sars-escape network for escape prediction of SARS-COV-2; Corona virus  
This repository contains codebase, links to dataset, and pretrained models for the paper "SARS-ESCAPE NETWORK FOR ESCAPE PREDICTION OF SARS-COV-2" . SARS-ESCAPE NETWORK is a Machine learning-based computational approach that predicts the escape protein sequences of SARS-COV-2 using sequence data alone.

Reference to Paper (Sars-escape network for escape prediction of SARS-COV-2) : https://doi.org/10.1093/bib/bbad140

## Dataset
All the intermediate datasets and final datasets for the training and evaluation are 
provided at zenodo doi: https://doi.org/10.5281/zenodo.7142638 

## Dependencies
The major packages of Python and tested versions are provided in env.yml file. 
All the dependencies can be installed from env.yml file using conda.

The experiments were run with Python version 3.7.0 on Ubnunu 18.04. 
## Experiments
The results of our experiments can be found in the model/ directory.
The main model is saved into model/integrated_model directory. The AUCs of respective datasets can 
be computed using below command line utilities

### Greaney dataset 
```bash
  python FE_Lang_Embed_Trainer_Full_Dataset.py --predict greaney
```
### Validation dataset 
```bash
  python FE_Lang_Embed_Trainer_Full_Dataset.py --predict validation
```
### Baum dataset 
```bash
  python FE_Lang_Embed_Trainer_Full_Dataset.py --predict baum
```

### Escape Prediction 
```bash
   python FE_Lang_Embed_Trainer_Full_Dataset.py --escape your_escape_seq 

```
Examples 

python FE_Lang_Embed_Trainer_Full_Dataset.py --escape FASVYAWNRKAISNCVADYS  (output = 1)

python FE_Lang_Embed_Trainer_Full_Dataset.py --escape QDKNTQEVFAYVKQIYKTPP  (output = 0) 

Note that escape window size must be of length 20

### Mutant Prediction 
```bash
   python FE_Lang_Embed_Trainer_Full_Dataset.py --mutant your_single_res_mutant 
```
Example: python FE_Lang_Embed_Trainer_Full_Dataset.py --mutant  Q497R

## Citation 
Prem Singh Bist, Hilal Tayara, Kil To Chong, Sars-escape network for escape prediction of SARS-COV-2, Briefings in Bioinformatics, Volume 24, Issue 3, May 2023, bbad140, https://doi.org/10.1093/bib/bbad140
