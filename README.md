# Design of Features to predict Hypotension during a Surgery


# Introduction

Cholescystectomy (gall bladder removal) is a common minimally invasive surgery performed under general anaesthesia. Nearly 50% patients undergo hypotension, i.e. a long episode of low arterial blood pressure, during this surgery within 20 min after the anaesthesia induction [1,2]. 

Hypotenstion is a critical event as it is linked to organ damage and postoperative complication. Early detection and management of hypotension can improve patient recovery and outcomes [3]. 

The goal of this project is to develop a classifier to predict the onset of a hypotensive episode 1 minute before it occurs. Emphasis is placed on the feature engineering, i.e. creating handcrafted features that summarize 20-second waveforms measured from the patient during the surgery.

The projects accounts for a total of 30 points in your final grade in this module: 
- 20 points based on the implementation you submit
- 10 points based on your presentation and its valuation are part of your grade within the final oral exam

# Dataset

The dataset was sourced from the VitalDB database [4]. It includes recordings from 58 cholecystectomy procedures (46 laparoscopies, 12 open surgeries).
The dataset is divided into : 
- a training set comprising 80% of the surgeries resulting in 10,253 non-overlapping windows of 20 seconds each
- a test set comprising 20% of the surgeries resulting in 1576 non-overlapping windows of 20 seconds each.

Each sample of the dataset comprises 20 seconds of five waveforms all sampled at 250 Hz:  
- Arterial Blood Pressure (measured invasively)
- Electrocardiogram, lead II
- Electrocardiogram, lead V5
- Photoplethysmogram  
- Capnogram

https://pubmed.ncbi.nlm.nih.gov/32768053/ (Wavelet decomp of ECG for Hypotension prediction)
https://arxiv.org/abs/2311.01142?utm_source=chatgpt.com (Empirical Mode Decomposition of ECG for Hypotension prediction)

# To many Features are not Goood!!!!!!
## When adding the default features of all singals, we get way worse
-> So, we start by finding out which signal brings most with default features


# TODO: The features we want to test:
- ECG lead II and V5
  - Wavelet Decomposition coefficients, entropy
  - Empirical Mode Decomposition (EMD) features

```python
# Wavelet Decomposition -> Entropy
coeffs = pywt.wavedec(signal, 'bior3.3', level=5)

features = []
for subband in coeffs:
  # Sample entropy
  sampen = ant.sample_entropy(subband)
  # Wavelet entropy (Shannon entropy of normalized coefficient energy)
  e = subband**2
  p = e / np.sum(e)
  wavelet_entropy = -np.sum(p * np.log2(p + 1e-12))
  features.extend([sampen, wavelet_entropy])
```

```python
emd = EMD()
imfs = emd(signal)

features = []
for imf in imfs[:5]:  # first 5 IMFs
    sampen = ant.sample_entropy(imf)
    # Additional features—fractal dimension, etc.
    features.append(sampen)
```

- Capnogram:
  - After some reasearch, there is no large correlation between capnogram and hypotension.
  - Since hypertension is also seen in the co2 value: “If cardiac output … is decreased … this is reflected in a decreased expired amount of CO₂.” (wikipedia)
  - We take the max End tidal CO2 value as a feature (mean and std)
- Arterial Blood Pressure
  - Systolic, Diastolic, Mean Arterial Pressure (MAP)
  - Pulse Pressure (PP)
  - Augmentation Index (AIx)
  - Heart Rate (HR)
  - Skewness, Kurtosis
  - Power Spectral Density (PSD) features in different frequency bands (e.g., LF, HF)
  - Wavelet Transform coefficients

- Electrocardiogram (ECG) lead II and V5
  - Heart Rate Variability (HRV) features (e.g., SDNN, RMSSD, pNN50)
  - Mean and Std of RR intervals
  - Skewness, Kurtosis
  - Power Spectral Density (PSD) features in different frequency bands (e.g., LF, HF)
  - Wavelet Transform coefficients

- Photoplethysmogram (PPG)
    - Pulse Rate Variability (PRV) features (e.g., SDNN, RMSSD, pNN50)
    - Mean and Std of peak-to-peak intervals
    - Skewness, Kurtosis
    - Power Spectral Density (PSD) features in different frequency bands (e.g., LF, HF)
    - Wavelet Transform coefficients
# Goals
Using the provided input signals,your task is to design features that will serve as inputs for training a machine learning model. For each 20-seconds segment, the model aims to predict whether an hypotensive episode will occur during a subsequent 20-second segment starting 1 minute later.


To do so, you should edit the script _features\_comp.py_ where you will create functions
that generate features which can then serve to build a dataset. Each feature you create
should be a function.

Each function computing a feature based on the input takes as argument a matrix in the form
$(time) \times (signal\ type)$, i.e. each column of the matrix corresponds to one input channel of the 
signal. 

Those features serve to build a dataset $X$ in a form $sample \times features$ that serves
as an input for a machine learning pipeline. 



# Submission and Assessment

## Deliverable

On Digicampus, should upload submit a .zip folder containing your code and requirements.txt based on the current repository. 

This must be done **48 hours before your personal exam** at the latest.
You should also **UPLOAD YOUR SLIDES (source and a pdf version) WITHIN THE ZIP FOLDER** as a backup.

Do not forget to put in the _authors.txt_ file your Name, Firstname, Mtk, and uni augsburg email.

If you use additional extra libraries you should put them in the _requirements\.txt_ file. 

In order to assess your model, we will create a virtual environment using the libraries you specified in the _requirements\.txt_.



## Assessment
We will generate your model by running the script _main.py_. 
This script prepares the datasets based on the features you generated, trains an extreme gradient boosting classifier on the train set, saves this model, and then tests this classifier on a validation set. 

A classification report providing the area under the receiver operating characteristic (AUROC) and precision-recall curve (AUPRC) are given. The AUPRC is the main metric you should consider optimizing.
During your final presentation, you should report at least AUROC and AUPRC obtained by your classifier on the test set.

Confusion matrix on the test set and additional classification metrics are also provided. 

As the focus of the project is on the feature generation, you should **NOT** touch the machine learning
pipeline. 

In your presentation, you should include **one slide** which presents the results generated by your **_main.py_** script.

We will then assess your model on another unseen dataset using the _test.py_ program. You can tryout this script to see if your code works, but it will be ultimately run on another dataset than the
one that is shared with you. 

If we are not able to run your model on the test set, there will be an automatic deduction of 5 points from your grade on the project implementation.

### For testing your code

To test if your model is likely to run, you may try running the following commands in a terminal assuming python 3.12.10 is installed on your computer.
Assuming you are at the top of your project folder. 
```

python -m venv .venv
.\.venv\Scripts\activate # If you are on windows
pip install -r requirements.txt
python main.py
python test.py
```
Alternatively, you can create a virtual environment via Venv using Vscode and select it as python interpreter. At the creation install the packages from the **_requirements.txt_** file.
You can then run the two scripts **_main.py_** and **_test.py_**.


## Presentation

The presentation during the oral exam should last 5 min. A question and answer part will follow this presentation. Considering the duration of the presentation, the following points should
be considered.

- You do not need to make an introduction of the project or the context.
- The structure of the presentation should be Methods, Results, Discussion/Conclusion.
- The Methods should focus on the signals you decided to use, the features you generated, and the signal processing techniques you applied to generate your features from the provided waveforms. 
- The Results should consist in one slide presenting the metrics your model achieved by running the _main.py_ script. 


# References
1. Gregory A, Stapelfeldt WH, Khanna AK, Smischney NJ, Boero IJ, Chen Q, et al. Intraoperative Hypotension Is Associated With Adverse Clinical Outcomes After Noncardiac Surgery. Anesthesia & Analgesia 2021;132(6). 

1. Lee J, Woo J, Kang AR, Jeong YS, Jung W, Lee M, et al. Comparative Analysis on Machine Learning and Deep Learning to Predict Post-Induction Hypotension. Sensors. 2020;20(16). 

1. Lee S, Lee HC, Chu YS, Song SW, Ahn GJ, Lee H, et al. Deep learning models for the prediction of intraoperative hypotension. British Journal of Anaesthesia. 2021 Apr 1;126(4):808–17. 

1. Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Scientific Data. 2022 Jun 8;9(1):279. 

,