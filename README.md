# GCWdvMSFC
Scripts and modules for training and testing deep neural networks that conducts Graphitized Cathode Wear Diagnosis via Multi-Source Feature Coupling.
Companion code to the paper "Cathodes of aluminum electrolysis state diagnosis by multi-source feature coupling based deep learning integrated with environmental knowledge".

<!-- https://www.xxxxxxxxx.com/

--------------------

Citation:
```
Authors et al. Cathodes of aluminum electrolysis state diagnosis by multi-source feature coupling based deep learning integrated with environmental knowledge.
journal. doi
```

Bibtex:
```
@article{,
  title = {Cathodes of aluminum electrolysis state diagnosis by multi-source feature coupling based deep learning integrated with environmental knowledge},
  author = {},
  year = {},
  volume = {},
  pages = {},
  doi = {},
  journal = {},
  number = {}
}
```
-------------------- -->

## Abstract information

Early diagnosis of the wear type and risk degree of aluminum electrolytic cell cathode is very necessary for the orderly production and safety prevention and control of aluminum electrolytic enterprises.we propose a method for diagnosing aluminum electrolysis cell cathode wear based on deep learning, termed Graphitized Cathode Wear Diagnosis via Multi-Source Feature Coupling (GCWdvMSFC). GCWdvMSFC was developed based on 127,391 sets of cathode monitoring data and achieved a PRC-AUC of 0.957 and a precision of 0.968 on a real-world test set containing 10,034 samples. Additionally, on two external test sets with 5,648 and 7,322 samples, the model achieved overall PRC-AUCs of 0.906 and 0.901, and precision scores of 0.933 and 0.890, respectively. Notably, GCWdvMSFC outperformed aluminum electrolysis safety management experts in diagnosing the state of graphitized cathodes. Our research directly impacts the study of cathode state diagnostics using multi-source coupled features and demonstrates the broader potential of such features in enhancing the diagnosis and management of aluminum electrolysis cell cathodes.

--------------------
## Requirements

This code was tested on Python 3 with Tensorflow `2.6.0`

In addition, the packages we are calling now is as follows:
- [x] tensorflow2.0     
- [x] sklearn
- [x] random
- [x] scipy
- [x] pandas
- [x] numpy
- [x] tabnet
- [x] tensorflow_addons  

## Framework illustration

- **input**: `shape = (N, 5000, 9)`. The input tensor, a signal of 10 seconds should contain the 5000 points of the ECG tracings sampled at 500Hz both in the training and in the test set. The last dimension of the tensor contains points of the 9 different leads. The leads are ordered in the following order: `{I, II, III, AVR, AVL, AVF, V1, V3, V5}`. All signal are preprocessed with noise removal techniques before feeding it to the neural network model. 
![example](https://github.com/shuaih720/CHDdECG/blob/main/Figures/ECG%20example.png)
- **framework illustration**: ``GCWdvMSFC.py``: Auxiliary module that defines the architecture of the deep neural network. The internal module structure is in the following files：``layers_BiLSTM-CNN.py``,``layers_Muilt-Head attention.py``,``layers_CNN with ICBAM.py``,``CRF with TransR.py``，``layers_AAW.py``.
![example1](https://github.com/shuaih720/CHDdECG/blob/main/Figures/An%20illustration%20of%20the%20deep%20learning%20based%20model.png)
- **train and test**: ``main.py``:Script for training the neural network and generating the neural network predictions on a given dataset.
- **output**: `shape = (N, 2)`. Each entry contains a probability between 0 and 1, and can be understood as the probability of a given abnormality to be present.

## Install from Github
```python
python
>>> git clone https://github.com/jipeng08/GCWdvMSFC
>>> cd GCWdvMSFC
>>> python setup.py install
```
(Typical install time on a "normal" desktop computer: very variable)

## Instructions for use
```python
python
>>> cd GCWdvMSFC
>>> python import.py
>>> python layers_BiLSTM-CNN.py
>>> python layers_Muilt-Head attention.py
>>> python layers_CNN with ICBAM.py
>>> python CRF with TransR.py
>>> python layers_AAW.py
>>> python GCWdvMSFC.py
>>> python main.py
```
OR running the integrated version 
```python
python
>>> cd GCWdvMSFC
>>> python Merged_GCW.py
```
Training the neural network and generating the neural network predictions on given datasets.
## License

This project is covered under the Apache 2.0 License.