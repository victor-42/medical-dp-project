# %%
'''
DO NOT EDIT THIS SCRIPT

Implement the pipeline for preparing the dataset, computing features, 
fitting and evaluating the machine learning model. 

Run this script to obtain your results on the validation set. 
Results obtained via this scripts are to be reported during the oral exam.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import VitalDBReader
from features_comp import prepare_data
from model import fit_model,evaluate_model,save_model

Xtrain,ytrain,Xval,yval=VitalDBReader.get_data("cholecystectomy",100)

Xtrain=prepare_data(Xtrain)
Xval=prepare_data(Xval)

estimator=fit_model(Xtrain,ytrain)

evaluate_model(estimator,Xval,yval)
save_model(estimator,'model.pkl')


# %%
