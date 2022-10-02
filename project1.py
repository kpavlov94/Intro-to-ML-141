import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show, boxplot, xticks, subplot, yticks, hist, ylim
from scipy.linalg import svd

filename = 'C:/Users/Marijo/Desktop/Introduction to Machine Learning and Data Mining/Project/data.csv'
df = pd.read_csv(filename)

raw_data = df.values
cols = range(1, 10)

X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])

classLabels = raw_data[:,-1]
classNames = ['Negative', 'Positive']
classDict = dict(zip(classNames,range(len(classNames))))

obj_df = df.select_dtypes(include=['object']).copy()
obj_df["famhist"] = obj_df["famhist"].astype('category')
obj_df["famhist"] = obj_df["famhist"].cat.codes

K = obj_df.values

for i in range(len(K)):
    X[i:, 4] = K[i]

y = raw_data[:,-1] ##

X = X.astype(float)

N, M = X.shape
C = len(classNames)

##Boxplot for each attribute
#standardizing the data
y1 = X - np.ones((N, 1))*X.mean(0)
y1 = y1*(1/np.std(y1,0))
figure(figsize=(10,6))
boxplot(y1)
xticks(range(1,10),attributeNames)
show()

##Histogram for each attribute
figure(figsize=(14,10))
u = int(np.floor(np.sqrt(M))); v = int(np.ceil(float(M)/u))
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i],20)
    xlabel(attributeNames[i])
    ylim(0,N/2)    
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
            plot(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], '.')
            if m1==NumAtr-1:
                xlabel(attributeNames[Attributes[m2]])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[Attributes[m1]])
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

y = y.astype(int)

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
r = np.arange(1,M+1)
figure(figsize=(10,6))
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Heart Disease: PCA Component Coefficients')
plt.show()

print('PC1:')
print(V[:,0].T)

print('PC2:')
print(V[:,1].T)

print('PC3:')
print(V[:,2].T)

# Attribute standard deviations

figure(figsize=(11,7))
plt.bar(r, np.std(X, 0))
plt.xticks(r, attributeNames)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Heart disease: attribute standard deviations')

# Subtract the mean from the data
Y1 = X - np.ones((N, 1))*X.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2,0))
# Here we're utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=.4)
nrows=3
ncols=2
# for k in range(2): # Plot attribute coefficients in principal component space
 
plt.subplot(nrows, ncols,  3)
for att in range(V.shape[1]):
    plt.arrow(0,0, V[att,i], V[att,j])
    plt.text(V[att,i], V[att,j], attributeNames[att])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
plt.grid()
# Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
     np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.title('Attribute coefficients')
plt.axis('equal')
