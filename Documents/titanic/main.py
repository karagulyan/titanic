import subprocess

# subprocess.call('reset')

import numpy as np
from numpy import *
import csv
import pandas as pd
import subprocess
# import matplotlib
#import matplotlib.pyplot as ply

import numpy.matlib as matlib 
import scipy.io as sio

import math
import random 


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import Ridge
#from sklearn.cross_validation import KFold
from cvxopt import matrix, solvers
from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp
from sklearn.decomposition import PCA
from itertools import product
from numpy import linalg as LA



X0 = pd.read_csv('train.csv', sep=',', header=None)
N = X0.shape[0]
print(N)

le_list = [1,2,4,5,6,7,9,11]
Xnew = X0.iloc[range(N)][le_list]
Xnew = pd.DataFrame.dropna(Xnew)

X_surv =  Xnew.iloc[range(N)][1]
X_people =  Xnew.iloc[range(N)][2,4,5,6,7,9,11]

N = Xnew.shape[0]

print(Xnew)

print(X_people)


# reg  = linear_model.LinearRegression()





print(Xnew)



# X0 = np.squeeze(np.array(X0))
# X = np.concatenate((X0, X1, X2))
# n = X.shape[0]

