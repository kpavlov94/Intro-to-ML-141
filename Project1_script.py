import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show, boxplot, xticks, subplot, yticks, hist, ylim
from scipy.linalg import svd

filename = 'data.csv'
df = pd.read_csv(filename)

raw_data = df.values  
cols = range(1, 11) 

classLabelss = raw_data[:,5]
classNamess = sorted(set(classLabelss))
classDicts = dict(zip(classNamess, range(2)))
raw_data[:,5] = np.asarray([classDicts[value] for value in classLabelss])

X = raw_data[:, cols]
X = X.astype(float)

origin = np.array(X[:, 4], dtype=int).T
K = origin.max()+1
origin_encoding = np.zeros((origin.size, K))
origin_encoding[np.arange(origin.size), origin] = 1
X = np.concatenate((X[:, :-1], origin_encoding),axis=1)

attributeNames = np.asarray(df.columns[cols])

classLabels = raw_data[:,-1]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape
C = len(classNames)

Xn = np.delete(X,4,1)
B, D = Xn.shape
attributeNamesXn=np.delete(attributeNames,[4,9])
attributeNamesXn=np.append(attributeNamesXn,["famhist_A","famhist_P"])

##Boxplot for each attribute
#standardizing the data
y1 = Xn - np.ones((B, 1))*Xn.mean(0)
y1 = y1*(1/np.std(y1,0))
figure(figsize=(9,9))
boxplot(y1)
xticks(range(1,11),attributeNamesXn)
show()

##Histogram for each attribute
figure(figsize=(14,14))
u = int(np.floor(np.sqrt(D))); v = int(np.ceil(float(D)/u))
for i in range(D):
    subplot(u,v,i+1)
    hist(Xn[:,i],20)
    xlabel(attributeNamesXn[i])
    ylim(0,B/2)    
show()

##Correlations between attributes
Attributes = [0,1,2,3,4,5,6,7]
NumAtr = len(Attributes)

figure(figsize=(12,12))
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(Xn[class_mask,Attributes[m2]], Xn[class_mask,Attributes[m1]], '.')
            if m1==NumAtr-1:
                xlabel(attributeNamesXn[Attributes[m2]])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNamesXn[Attributes[m1]])
            else:
                yticks([])
legend(classNames)
show()

##Computing the PCA of the Data and 
##plotting the percent of variance explained by the principal components as well as the cumulative variance

# PCA by computing SVD of Y
U,S,Vh = svd(y1,full_matrices=False)
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 
threshold = 0.9

plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

##Plotting principal component 1 and 2 against each other in a scatterplot
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    
# Project the centered data onto principal component space
Z = y1 @ V
# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Heart Disease: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
show()

##the data projected onto the considered principal components
#the first 3 components explaiend more than 90 percent of the variance. 
#Let's look at their coefficients:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,D+1)
figure(figsize=(10,10))
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNamesXn)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Heart Disease: PCA Component Coefficients')
plt.show()
