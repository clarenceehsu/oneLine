"""
oneline.machinelearning is the set of functions and features for machine learning and extended deep
learning, which contains:

    1. AverageMeter, a average meter for counting and analyze the data during training
    2. NeuralNetwork, extended deep learning module for fast training
    3. MachineLearning, a set of useful classical machine learning methods
    4. MachineLearningModel, a container for machine learning training
"""

from .average_meter import AverageMeter
from .neural_network import NeuralNetwork
from .machine_learning import *
