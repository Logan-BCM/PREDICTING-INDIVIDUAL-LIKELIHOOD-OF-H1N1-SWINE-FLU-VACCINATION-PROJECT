## PROJECT TITLE:
***

PREDICTING INDIVIDUAL LIKELIHOOD OF H1N1 "SWINE FLU" VACCINATION USING LOGISTIC REGRESSION AND DECISION TREE CLASSIFIERS


## DESCRIPTION:
***
This project aims to analyze vaccination behavior and identify key factors influencing the uptake of the H1N1 and seasonal flu vaccines. Using a dataset containing demographic, behavioral, and opinion-based features, the project leverages both exploratory data analysis (EDA) and machine learning models to uncover insights and provide actionable recommendations for public health stakeholders.

Through EDA, trends and patterns related to vaccination behavior were identified, highlighting critical subgroups and potential barriers to vaccine acceptance. 
Machine learning classifiers, including logistic regression and decision tree models, were developed and tuned to predict vaccination likelihood and prioritize features driving these decisions.

The findings and recommendations derived from this analysis are intended to support public health organizations in designing targeted, data-driven campaigns to improve vaccination rates, particularly among under-vaccinated populations. This project serves as a valuable resource for public health efforts aiming to mitigate the spread of infectious diseases through increased vaccine adoption.


## TOOLS AND TECHNOLOGIES USED:
***

Standard Data Science packages: 

import pandas as pd, numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,auc
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve,ConfusionMatrixDisplay
from sklearn import tree
%matplotlib inline


## DATASETS:
***

In this repo there is a file (Dataset) that contains the datasets used in this project 


## ADDITIONAL RESOURCES

In this repo there are some additional images that have been used in the notebook.

## PROJECT REPORT:
***

There is also a PDF file in the repo that is a report of the entire project.



