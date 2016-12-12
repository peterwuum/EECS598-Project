# EECS598 Project: Topic Models with Network Regularization 
# Authors
 - Zheng Wu
 - Wei-Hsin Chen
 - Yuqi Gu
 - Xuefei Zhang

# Introduction

In this project, we model the text generating process in a large corpus with network structure through a joint model of PLSA and network regularization. Our contributions include the following things. Firstly, we implement the classical PLSA algorithm on the Microsoft Academic Graph as a baseline. We test its performance on different proportions of the entire dataset, each of which is randomly sampled from the entire graph. Then we implement NetPLSA model and also vary the parameter settings to compare the results. Then we use several methods to evaluate the NetPLSA algorithm, including document classification analysis and topic interpretability analysis. We compare the results to baseline approaches, the PLSA model and the spectral clustering model respectively.


# Getting Started

## Prerequisites
The dependent python libraries for this project are listed as follow:
 - numpy
 - scipy
 - sklearn
 - nltk
 - click
 - statistics
 - multiprocessing
 - matplotlib

## Installing
Please run the following in terminal before running the code
```
pip install -r requirements.txt
```

## Running Example
### Get Data From Microsoft Academic Graph
```
python DataProcessingPipeline.py
```

### Fit NetPLSA Model or PLSA model on Selected Academic Graph
```
python netplsa_with_plsa.py
```

### Evaluate NetPLSA Model or PLSA Model
```
python evaluate.py
```

# Function Description
 - DataProcessing.py: Select papers from Microsoft Academic Graph
 - netplsa_with_plsa.py: Run NetPLSA Model (network = True) or PLSA model (network = False)
 - evaluate.py: Classification, Word Intrusion, Network Clustering Tasks on Topic Model
 - networkclustering.py: Cluster papers based on network structure
 - overlapNMI.py: Calculate overlap NMI score based on true labels and predict labels (differ from NMI score function in sklearn library)
 - word_intrusion.py: Evaluate topic interpretability

# Pre-selected Dataset
The files in PROCESSED folder are the data we used in our project

# License
This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the LICENSE.md file for details

