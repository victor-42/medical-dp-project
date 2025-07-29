# %%
'''
DO NOT EDIT THIS SCRIPT

Implement the pipeline for preparing the dataset, computing features, 
fitting and evaluating the machine learning model. 

This script will be used to test your data on a hidden dataset. 

Some parameters have been fixed to provide you an example to test if 
your code is likely to work.

Results obtained via this script should not be reported by you during the
oral exam as the example dataset here is not representative of the data.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import VitalDBReader
from features_comp import prepare_data
from model import evaluate_model,load_model

Xtrain,ytrain,Xval,yval=VitalDBReader.get_data("cholecystectomy",10)

Xtest,ytest=np.concat([Xtrain,Xval],axis=0),np.concat([ytrain,yval],axis=0)

Xtest=prepare_data(Xtest)

estimator=load_model('model.pkl')
evaluate_model(estimator,Xtest,ytest)

