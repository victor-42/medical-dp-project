import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from xgboost import plot_importance

from dataset import VitalDBReader
from features_comp import prepare_data
from model import fit_model,evaluate_model,save_model

import pickle

if __name__ == "__main__":
    # Chekc if pickle file exists, otherwise create it
    pickle_path = 'data_quicksafe.pkl'

    try:
        with open(pickle_path, 'rb') as f:
            Xtrain, ytrain, Xval, yval = pickle.load(f)
        print("Data loaded from pickle file.")
    except FileNotFoundError:


        Xtrain,ytrain,Xval,yval=VitalDBReader.get_data("cholecystectomy",999)

        with open(pickle_path, 'wb') as f:
            pickle.dump((Xtrain, ytrain, Xval, yval), f)

    to_dispose = [1885, 3427, 3546, 5074, 6924, 7103, 7102, 8140]
    Xtrain = np.delete(Xtrain, to_dispose, axis=0)
    ytrain = np.delete(ytrain, to_dispose, axis=0)

    to_dispose_val = [916]
    Xval = np.delete(Xval, to_dispose_val, axis=0)
    yval = np.delete(yval, to_dispose_val, axis=0)
    """
    def compute_stats(x,column=0):
        #compute some descriptive stats
        stats=[]
        mean,std,skw,kurt=np.mean(x[:,column]),np.std(x[:,column]),skew(x[:,column]),kurtosis(x[:,column])
        stats+=[mean,std,skw,kurt]
        return stats

    def compute_features(x):
        features=[]
        #call a function you programmed above to compute some features on a signal
        features=compute_stats(x,column=0) #here just an example
        return features

    def prepare_data(x):
        X=np.stack([compute_features(x_) for x_ in x])
        return X
    """

    Xtrain=prepare_data(Xtrain, )
    Xval=prepare_data(Xval, )
    estimator=fit_model(Xtrain,ytrain)


    evaluate_model(estimator,Xval,yval)
    save_model(estimator,'model.pkl')


