#######################################################

	 				#graphic tools

#######################################################

import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

import numpy as np
import math

import pandas as pd

import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



#function wich plots the histogram distribution of several variables
#maximum of six var per raw
#vars is list of variables not necessarily from the same df.
#names is the list of var names
#function only useful if the vars are not in the same df. Otherwise better use df.hist() from pandas
def plot_distribution(vars, names, bins=50, color="firebrick"):

	p=len(vars)
	if p<=4:
		fig, ax=plt.subplots(1,p, figsize=(20,5))
		for k in range(len(vars)):
			ax[k].hist(vars[k], bins=bins, color=color)
			ax[k].set_xlabel(names[k])
	else:
		q=p//4+1
		fig, ax=plt.subplots(q,4, figsize=(20,15))
		for k in range(len(vars)):
			i=k//4 #raw
			j=k%4-1
			ax[i,j].hist(vars[k], bins=bins, color=color)
			ax[i,j].set_xlabel(names[k])
	plt.show()


#plot the distribution of numerical features by label
#label must be binary 
#column=the list of numerical var
#target = name of the target variable. Example: "y"
def plot_distribution_class(data, target,  column=None,  bins=50, color1="firebrick", color2="blue", alpha1=0.5, alpha2=0.3):
	subset0=data[data[target]==0]
	subset1=data[data[target]==1]

	p=len(column)
	n=math.ceil(p/4)
	fig, ax=plt.subplots(n,4, figsize=(20,15))
	ax=ax.ravel()
	
	for i in range(p):
		ax[i].hist(subset0[column[i]], bins=bins, color=color1, alpha=alpha1)
		ax[i].hist(subset1[column[i]], bins=bins, color=color2, alpha=alpha2)
		ax[i].set_title(column[i], fontsize=9)
		ax[i].set_yticks(())

	ax[0].legend(['0','1'],loc='best',fontsize=8)
	plt.tight_layout()
	plt.show()



#plot of counts for categorical variables (barplot)
#must be from the same data_frame
#if we want bar plots per group we can use the argument group
def plot_counts(columns, data, group=None): #col =list of categorigal name
	n=2
	m=math.ceil(len(columns)/n)
	fig, axes=plt.subplots(m,n, figsize=(15,10))
	axes=axes.flatten()

	for i, col in enumerate(columns):
		sns.countplot(x=col, data=data, hue=group, palette="rocket",  ax=axes[i])
		fig.tight_layout()

#box_plot for numerical columns
#must be from the same data_frame
def boxplot_vars(columns, data, orientation, color=None):
	n = 3
	m = math.ceil(len(columns)/n)
	fig, axes =plt.subplots(m,n, figsize=(15,5))
	axes = axes.flatten()

	for i,t in enumerate(columns):
		sns.boxplot(x=t, data=data, orient=orientation, ax=axes[i], color=color)
		fig.tight_layout()


#plot the pca transformation with one color per class (binary problem)
#W must be numeric
#y must be binary
def plot_pca_classification(X,y):
	pca = PCA(n_components=2)
	scaler=StandardScaler()
	X_scaled=scaler.fit_transform(X)

	principalComponents = pca.fit_transform(X_scaled)
	df= pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'], index=X.index)
	df["target"]=y

	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_title('2 component PCA', fontsize = 20)


	targets = [1,0]
	colors = ['r','b']
	for target, color in zip(targets,colors):
		indicesToKeep = df["target"] == target
		ax.scatter(df.loc[indicesToKeep, 'principal component 1'], df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
	ax.legend(targets)
	ax.grid()
	plt.show()

	print("percentage of explained variance", pca.explained_variance_ratio_)


#plot the correlation between the principal components and features
#W must be numeric
def pca_heat_map(X,p): #p=nb of principal components
	feature_names=X.columns.tolist()
	pca=PCA(n_components=p)
	scaler=StandardScaler()
	X=scaler.fit_transform(X)
	pca.fit_transform(X)

	plt.matshow(pca.components_,cmap='viridis')
	index=np.arange(p).tolist()
	plt.yticks(index ,["Comp_"+str(i+1) for i in index],fontsize=10)
	plt.colorbar()
	plt.xticks(range(len(feature_names)),feature_names,rotation=65,ha='left')
	plt.show()
