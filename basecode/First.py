import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import random

mat = loadmat('/home/inspire/Dropbox/UB/ML/Project/pa1/basecode/mnist_all.mat')
print mat.keys()