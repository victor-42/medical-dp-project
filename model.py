'''
DO NOT EDIT THIS SCRIPT !!!

Machine learning pipeline for the model assessment. 

The estimator consists in an XGBoost classifier applied on the
standardized data data projected via PCA.

A random-search over the hyperparameters of the XGBoost classifier 
is performed to maximize the AUROC. 

A classification threshodld for the classifier is adjusted to maximize
the f1-score on the train set. 

Data is then assessed on the test set via 
AUROC, AUPRC, Confusion matrix
'''
#%% xgboost
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_recall_curve, auc,classification_report,confusion_matrix
import matplotlib.pyplot as plt

from pickle import dump,load

SEED=42#for reproductibility

def fit_model(X, y):
    '''
    Fit an XGBoost classifier to a dataset. A randomized search optimizes the hyperparameters of the classifier by
    maximizing the 5-fold cross-validation AUROC. 

    Parameters
    ----------
    X: array-like
        2d array (sample, features) dataset. 
    y: array-like
        1d array (sample,) dataset binary target values to fit.
    '''
    xgb = XGBClassifier(max_depth=4, n_estimators=50, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',random_state=SEED)
    pipeline = Pipeline([
        ('standard_scaler', StandardScaler()), 
        # ('PCA',PCA()),
        ('model', xgb)
    ])

    param_grid = {
        'model__max_depth': [2, 3, 4, 5, 7, 10],
        'model__n_estimators': [10, 100, 500],
    }
    grid = RandomizedSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc_ovo',random_state=SEED)
    grid.fit(X, y)

    model=grid.best_estimator_

    return model

def calibrate(estimator, X, ytrue):
    '''
    Calibrate a binary classifier decision-threshold by maximizing the F1-score on a given 
    input dataset.

    Parameters
    ----------
    estimator: sklearn.base.Estimator
        An sklearn-compatible binary classifier implementing the predict_proba method.
    X: array-like
        2d array (sample, features) dataset on which the decision threshold should be calibrated. 
    y: array-like
        1d array (sample,) dataset binary target values to fit.

    Returns
    -------
    best_threshold: float
        Decision threshold for the binary classifier
    '''
    y_pred_proba = estimator.predict_proba(X)[:,-1]
    precisions,recalls,thresholds=precision_recall_curve(ytrue,y_pred_proba)
    fscore=2*precisions*recalls/(precisions+recalls)
    best_threshold=thresholds[np.nanargmax(fscore)]
    return best_threshold

def evaluate_model(model, X,y, threshold=0.5):
    '''
    Function to assess the performance of a model on a test set. Displays and plots results
    on figures. 

    Parameters
    ----------
    model: sklearn.base.Estimator
        A binary classifier to assess
    X: array-like
        2d array (sample, features) dataset on which the `model` is assessed. 
    y: array-like
        1d array (sample,) true labels associated to the dataset `X`.
    '''
    yproba = model.predict_proba(X)[:, 1]

    auroc_gbm = roc_auc_score(y, yproba)
    test_prc, test_rec, thresholds = precision_recall_curve(y, yproba)
    test_auprc = auc(test_rec, test_prc)
    fpr_gbm, tpr_gbm, _ = roc_curve(y, yproba)
    print(f'GBM auroc: {auroc_gbm:.3f}, auprc: {test_auprc:.3f}', flush=True)

    plt.figure(figsize=(5,5))
    plt.plot(fpr_gbm, tpr_gbm, label='GBM = {:0.3f}'.format(auroc_gbm))
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Receiver Operat Characteristic')

    plt.figure(figsize=(5,5))
    plt.plot(test_rec, test_prc, label='AUPRC = {:0.3f}'.format(test_auprc))
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Precision-Recall Curve')


    print(classification_report(y,(yproba>threshold)))
    print(confusion_matrix(y,(yproba>threshold)))
    return 

def save_model(estimator, path):
    '''
    Dump a machine learning model into a pickle file.

    Parameters
    ---------
    estimator: sklearn estimator
        The estimator to save
    
    path: str
        The path of the file where the model should be dumped
    '''
    with open(path,'wb') as f: 
        dump(estimator,f,protocol=5)
    return
def load_model(path):
    '''
    Dump a machine learning model into a pickle file.

    Parameters
    ---------   
    path: str
        The path of the file where the model should be dumped
    
    Returns
    -------
    estimator: sklearn estimator
        The estimator loaded from the pickle file
    '''
    with open(path,'rb') as f:
        clf=load(f)
    return clf