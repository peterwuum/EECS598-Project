# EECS598 - Project
## Authors:
 - Zheng Wu
 - Wei-Hsin Chen
 - Yuqi Gu
 - Xuefei Zhang

## Introduction:


## Getting Started

# Prerequisites
The dependent python libraries for this project are listed as follow:
 - numpy
 - scipy
 - sklearn
 - nltk
 - click
 - statistics
 - multiprocessing
 - matplotlib

# Installing
Please run the following in terminal before running the code
```
pip install -r requirements.txt
```

## Running Example
Get Data From Microsoft Academic Graph
```
python DataProcessingPipeline.py
```

Fit NetPLSA Model or PLSA model on Selected Academic Graph
```
python netplsa_with_plsa.
```

Evaluate NetPLSA Model or PLSA Model
```
python evaluate.py
```

## Function Description
 - DataProcessing.py: Select papers from Microsoft Academic Graph
 - netplsa_with_plsa.py: Run NetPLSA Model (network = True) or PLSA model (network = False)
 - evaluate.py: Classification and Word Intrusion Task on Topic Model
 - networkclustering.py: Cluster papers based on network structure
 - overlapNMI.py: Calculate overlap NMI score based on true labels and predict labels (differ from NMI score function in sklearn library)

## Pre-selected Dataset
The files in PROCESSED folder are the data we used in our project

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the LICENSE.md file for details

