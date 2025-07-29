'''
Edit this script to create functions that compute features based on the input signals.
For each function you create to compute a feature, put a docstring describing the feature
you compute. The docstring can be written in the same style as in the present module. 
'''
import numpy as np
from scipy.stats import kurtosis,skew


def compute_stats(x,column=0):
    '''Example of feature computation
    
    Parameters
    ----------
    x: array_like
        2d Input signal where the axis 0 corresponds to the time whereas each column corresponds to 
        an input channel
    column: int
        The column of `x` that should be processed by the function

    Returns
    -------
    output: array_like
        A 1-d array containing features computed by the function over the specified `column` of `x`.
    '''
    #compute some descriptive stats
    stats=[]
    mean,std,skw,kurt=np.mean(x[:,column]),np.std(x[:,column]),skew(x[:,column]),kurtosis(x[:,column])
    stats+=[mean,std,skw,kurt]

    output=np.array(stats)
    return output

def compute_features(x):
    '''
    Compute the features to describe a recording.

    Parameters
    ----------
    x: array_like
        2d Input signal where the axis 0 corresponds to the time whereas each column corresponds to 
        an input channel

    Returns
    -------
    output: array_like
        A 1-d array concatenating all the features you computed from the signal.

    '''

    #EDIT YOUR CODE HERE TO COMPUTE FEATURES
    features=[]
    #call a function you programmed above to compute some features on a signal
    features=compute_stats(x,column=0) #here just an example

    return features
def prepare_data(x):
    '''
    Apply compute_features to the data input tensor.

    Parameters
    ----------
    x: array_like
        A 3-d array of waveform signals where the first axis corresponds to the input record, the
        second axis to the time, and the third axis to the input channel of the record.

    Returns
    -------
    X: array_like
        A 2-d array dataset in which the first axis (lines) corresponds to the input recording
        and the second axis (columns) corresponds to the recording features computed with `compute_features`
    '''
    X=np.stack([compute_features(x_) for x_ in x])
    return X

if __name__=='__main__':
    print('running features_comp.py')