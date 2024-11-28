import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, Attention, Dense, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention, Add, Flatten, LayerNormalization
from tensorflow.keras.layers import Input, Conv1D, Multiply, Add, GlobalAveragePooling1D, Dense, Reshape, Activation
from tensorflow.keras.layers import GlobalMaxPooling1D, Concatenate, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, Multiply, Concatenate, GlobalAveragePooling1D
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pykeen
from pykeen.pipeline import pipeline
from pykeen.models import TransR
from pykeen.triples import TriplesFactory
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, MultiHeadAttention, Add, Multiply, Concatenate, GlobalAveragePooling1D, Dense, LayerNormalization, Flatten, TimeDistributed
from tensorflow.keras.layers import Embedding
import tensorflow_addons as tfa
import os
import sys
from setuptools import find_packages, setup
from setuptools.command.install import install
from subprocess import call, check_output
from sys import platform
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
import math