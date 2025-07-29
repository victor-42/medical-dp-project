'''
DO NOT EDIT THIS SCRIPT !!!

This script selects cases from Vitaldb database and prepares a train and test set
'''
import numpy as np
import pandas as pd
import vitaldb

def diff(x):
    x[1:] -= x[:-1]
    return x

class VitalDBReader: 
    '''
    class to prepare the dataset using VitalDB API.
    Returns dataset arrays in the form of a 3-d array: 
    (recording, time, channel)

    Each recording contains 4 channels: arterial blood pressure, ECG lead II,ECG lead V5, PPG, Capnography

    '''
    def __init__(self,surgery, max_cases=50, tracklist='https://api.vitaldb.net/trks',case_info="https://api.vitaldb.net/cases"):
        '''
        Constructor

        Parameters
        ----------
        max_cases: maximum number of recordings allowed to be added to the dataset.
        
        '''
        self.tracklist=tracklist
        self.case_info=case_info
        self.surgery=surgery
        
        self.MINUTES_AHEAD = 1  # Predict hypotension 1 minutes ahead
        self.MAX_CASES = max_cases  # Maximum number of cases for this example
        self.SRATE = 250  # sampling rate for the arterial waveform signal Hz
        self.DURATION=20 # window duration s
        self.OVERLAPP=0# window overlapp
        self.test_size=0.2
        
        #included signals
        self.included_signals=['SNUADC/ART',
                               'SNUADC/ECG_II',
                               'SNUADC/ECG_V5',
                               'SNUADC/PLETH',
                               'Primus/CO2']
    def filter_cases(self):
        df_trks = pd.read_csv('https://api.vitaldb.net/trks')  # read track list
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # read case information
        caseids = list(
            set(df_trks[df_trks['tname'] == 'SNUADC/ART']['caseid']) &
            set(df_trks[df_trks['tname'] == 'SNUADC/ART']['caseid']) &
            set(df_trks[df_trks['tname'] == 'SNUADC/ECG_II']['caseid']) &
            set(df_trks[df_trks['tname'] == 'SNUADC/ECG_V5']['caseid']) &
            set(df_trks[df_trks['tname'] == 'SNUADC/PLETH']['caseid']) &
            set(df_trks[df_trks['tname'] == 'Primus/CO2']['caseid']) &
            set(df_cases[df_cases['age'] > 18]['caseid']) &
            set(df_cases[df_cases['age'] >= 18]['caseid']) & 
            set(df_cases[df_cases['weight'] >= 30]['caseid']) & 
            set(df_cases[df_cases['weight'] < 140]['caseid']) & 
            set(df_cases[df_cases['height'] >= 135]['caseid']) & 
            set(df_cases[df_cases['height'] < 200]['caseid']) & 
            set(df_cases[df_cases['opname'].str.contains(self.surgery, case=False)]['caseid'])&# 58 cases
            set(df_cases[df_cases['ane_type'] == 'General']['caseid'])
    )
        print('Total {} cases found'.format(len(caseids)))

        return caseids
    def check_signal_validity(self,segx,segy):
        # check the validity of this segment
        valid = True
        if np.isnan(segx[:,0]).mean() > 0.1:
            valid = False
        elif np.isnan(segy).mean() > 0.1:
            valid = False
        elif (segx[:,0] > 200).any():
            valid = False
        elif (segy > 200).any():
            valid = False
        elif (segx[:,0] < 30).any():
            valid = False
        elif (segy < 30).any():
            valid = False
        elif np.max(segx[:,0]) - np.min(segx[:,0]) < 30:
            valid = False
        elif np.max(segy) - np.min(segy) < 30:
            valid = False
        elif (np.abs(np.diff(segx[:,0])) > 30).any():  # abrupt change -> noise
            valid = False
        elif (np.abs(np.diff(segy)) > 30).any():  # abrupt change -> noise
            valid = False
        return valid
    
    def get_full_dataset(self):
        '''
        Construct a 3d dataset sourcing the VitalDB database.

        Return
        ------
        x: array_like
            3-d tensor of extracted segments from each recording
        y: array_like
            The extracted label for each segment
        valid_mask: array_like
            Mask selecting the valid segments from the recording using
            `check_signal_validity`
        c: array_like
            List of the case identifiers of each segment
        
        '''
        caseids=self.filter_cases()
        x = []  # input with shape of (segements, timepoints, channels)
        y = []  # output with shape of (segments)
        valid_mask = []  # validity of each segement
        c = []  # caseid of each segment

        # maximum number of cases
        for caseid in caseids:
            print(f'loading {caseid}', end='...', flush=True)

            # read the arterial waveform
            arts = vitaldb.load_case(caseid,self.included_signals, 
            1 / self.SRATE)

            case_sample = 0
            case_event = 0
            for i in range(0, len(arts) - self.SRATE * (self.DURATION + (1 + self.MINUTES_AHEAD) * 60), int((1-self.OVERLAPP)*self.DURATION)* self.SRATE):
                segx = arts[i:i + self.SRATE * self.DURATION,:]
                segy = arts[i + self.SRATE * (self.DURATION + self.MINUTES_AHEAD * 60):i + self.SRATE * (self.DURATION + (self.MINUTES_AHEAD + 1) * 60),0]

                # check the validity of this segment
                valid=self.check_signal_validity(segx,segy)

                # 2 sec moving avg
                n = 2 * self.SRATE  
                segy = np.nancumsum(segy, dtype=np.float32)
                segy[n:] = segy[n:] - segy[:-n]
                segy = segy[n - 1:] / n

                evt = np.nanmax(segy) < 65
                x.append(segx)
                y.append(evt)
                valid_mask.append(valid)
                c.append(caseid)
                
                if valid:
                    case_sample += 1
                    if evt:
                        case_event += 1

            if case_sample > 0:
                print("{} samples {} ({:.1f} %) events".format(case_sample, case_event, 100 * case_event / case_sample))
            else:
                print('no sample')

            if len(np.unique(c)) >= self.MAX_CASES:
                break

        # final caseids
        caseids = np.unique(c)

        # convert lists to numpy array
        x = np.array(x)
        y = np.array(y) 
        valid_mask = np.array(valid_mask)
        c = np.array(c)

        # forward filling
        for i in range(x.shape[0]):
            x[i] = pd.DataFrame(x[i]).ffill(axis=0).bfill(axis=0).values

        return x,y,valid_mask,c,caseids
    
    def split_dataset(self,c,caseids):
        ncase = len(caseids)
        ntest = int(ncase *self.test_size)
        ntrain = ncase - ntest
        caseids_train = caseids[:ntrain]
        caseids_test = caseids[ncase - ntest:ncase]
        ### splitting into train set and test set
        train_mask = np.isin(c, caseids_train)
        test_mask = np.isin(c, caseids_test) 
        return train_mask,test_mask
    
    def get_train_dataset(self,x,y,valid_mask,c,caseids):
        train_mask,_=self.split_dataset(c,caseids)
        train_x_valid = x[train_mask & valid_mask]
        train_y_valid = y[train_mask & valid_mask]
        return train_x_valid,train_y_valid

    def get_test_dataset(self,x,y,valid_mask,c,caseids):
        _,test_mask=self.split_dataset(c,caseids)
        test_x_valid = x[test_mask & valid_mask]
        test_y_valid = y[test_mask & valid_mask]
        return test_x_valid,test_y_valid
    
    @classmethod
    def get_data(cls,surgery,max_cases=50):
        '''
        Construct the train and test set.

        Parameters
        ----------
        max_cases: the maximum amount of recordings included in the dataset.
        surgery: str
            The surgery for which cases are retrieved
        Returns
        -------
        xtrain: array_like
            The samples of the training set
        ytrain: array_like
            The labels to be predicted on the training set
        xtest: array_like
            The samples of the test set
        ytest: array_like
            The labels to be predicted on the test set
        '''
        loader=cls(surgery,max_cases)
        x,y,valid_mask,c,caseids=loader.get_full_dataset()
        xtrain,ytrain=loader.get_train_dataset(x,y,valid_mask,c,caseids)
        xtest,ytest=loader.get_test_dataset(x,y,valid_mask,c,caseids)
        print('Included signals: ',loader.included_signals)
        print('Train set shape {}, test set shape {}'.format(str(xtrain.shape),str(xtest.shape)))
        print('Sample x Time x Channel')
        return xtrain,ytrain,xtest,ytest
        
    
if __name__=='__main__':
    Xtrain,ytrain,Xtest,ytest=VitalDBReader.get_data()
    print(Xtrain.shape,Xtest.shape)
    print(pd.DataFrame(ytrain).value_counts())
    print(pd.DataFrame(ytest).value_counts())