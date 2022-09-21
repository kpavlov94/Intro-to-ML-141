import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
from scipy.linalg import svd

filename = 'data.csv'

df = pd.read_csv(filename)

raw_data = df.values  
cols = range(0, 11) 
X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])

classLabels = raw_data[:,-1]

classNames = np.unique(classLabels)

classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape

C = len(classNames)

N1 = np.delete(X,5,1)
N2 = np.delete(N1,9,1)
print(N2)
B, D = N2.shape

y1 = N2 - np.ones((B, 1))*N2.mean(0)

# subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
y2 = N2 + np.ones((B, 1))*N2.mean(0)
y2 = y2*(1/np.std(y2,0))
# here were utilizing the broadcasting of a row vector to fit the dimensions 
# of y2

# store the two in a cell, so we can just loop over them:
# ys = [y1, y2]
# titles = ['zero-mean', 'zero-mean and unit variance']
# threshold = 0.9

