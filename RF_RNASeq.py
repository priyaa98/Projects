# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:28:58 2019

@author: Priyamvadha
"""
#Packages required
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,r2_score
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
from numpy import inf

#Import dataset from local folder
data= pd.read_csv("C:/Users/Priyamvadha/Desktop/My Files/Data sets/Gene expression cancer RNA-Seq Data/data.csv")

#Example Visulizations of the dataset
##Count Plot of Classes
ax=sns.countplot(x="Class", data=data, facecolor=(0,0,0,0), linewidth=5, edgecolor=sns.color_palette("Blues_d",2))
##Normal Plot (Scatter and Bar)
sns.pairplot(data.loc[:, data.columns !="Class"].iloc[:,1:5])
##Comparative Density-Contour Plot Gene 1 to Gene 5
g1 = sns.PairGrid(data.loc[:, data.columns !="Class"].iloc[:,1:5])
g1.map_diag(sns.kdeplot)
g1.map_offdiag(sns.kdeplot, n_levels=6)
##Single Density-Contour Plot Gene 1 VS Gene 2
g2 = sns.jointplot(x="gene_1", y="gene_2", data=data, kind="kde", color="k")
g2.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker=".")
g2.ax_joint.collections[0].set_alpha(0)

#Separating features and labels
y= data.Class
label=set(y)
x=data.loc[:, data.columns!="Class"]
labels=np.array(data["Class"])
labels=pd.get_dummies(labels)
features= np.array(x)
features_list=list(x.columns)

#Creating Training, Validation and Testing Sets
train_val_features, testing_features, train_val_labels, testing_labels = train_test_split(features, labels, test_size = 0.2, random_state = 3)
testing_train_features,val_features,testing_train_labels,val_labels= train_test_split(train_val_features, train_val_labels, 
                                                                                        test_size = 0.2, random_state = 3)

#Creating dummy Random Forest model to find best parameters
rf_dummy=RandomForestClassifier()

#Finding best parameters using RandomizedSearchCV
##Parameter choices
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
max_features = ["auto", "sqrt"]
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True]
criterion=["gini","entropy"]
oob_score=[True,False]
warm_start=[True,False]
class_weight=[None,"balanced"]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterion,
               'oob_score': oob_score,
               'warm_start': warm_start,
               'class_weight': class_weight
               }
##Testing for best parameters
rf_random = RandomizedSearchCV(estimator = rf_dummy, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, 
                               random_state=3, n_jobs = -1)
##Fitting for the training set
rf_random.fit(testing_train_features,testing_train_labels)
##Best parameters
print(rf_random.best_params_)

#Real Random Forest model with the best parameters applied
rf = RandomForestClassifier(n_estimators=277,min_samples_split=5,min_samples_leaf=1,max_features="auto",max_depth=420,
                            bootstrap=True,oob_score=True,verbose=1,warm_start=True,random_state=3,criterion="entropy",
                            class_weight="balanced",n_jobs=-1)
##Fitting Training Set
rf.fit(testing_train_features,testing_train_labels)
##Predicting for Validation Features Set
rf_predict = rf.predict(val_features)
##Soring for unseen Testing Set
test_score=rf.score(testing_features, testing_labels)
print(test_score)
##Out of the bag Score
oob_score=rf.oob_score_
print(oob_score)
##Top 10 important features
skplt.estimators.plot_feature_importances(rf, feature_names=list(data.columns[1:]), max_num_features=10, figsize=(12,12))
##R-2 Score
r_score=r2_score(val_labels,rf_predict,multioutput="variance_weighted")
print(r_score)
##Confusion matrox map
skplt.metrics.plot_confusion_matrix(val_labels.values.argmax(axis=1), rf_predict.argmax(axis=1),figsize=(12,12))
##Classification Report consisting of Precision, Recall and F1-score
class_rep=classification_report(val_labels, rf_predict,target_names=label)
print(class_rep)

#Finding Accuracy
errors=abs(rf_predict-val_labels)
mae=round(np.mean(errors),2)
mape= 100*(errors/val_labels)
mape[mape==inf]=0
np.nan_to_num(mape,copy=False)
accuracy=100-np.mean(mape)
percents= round(accuracy,2)
print(percents.mean())

#Extracting a tree for visualization
tree=rf.estimators_[10]
export_graphviz(tree,out_file="tree_sample.dot",feature_names=features_list, rounded=True, precision=1)
(graph, )=pydot.graph_from_dot_file("tree_sample.dot")
graph.write_png('tree_sample.png')
