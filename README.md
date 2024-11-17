# MATE-Net

## Introduction
This repository includes the implementation of Multimodal and ATtention-based Expansion Network (MATE-Net) for acute ischemic stroke (AIS) patients in 90 days mRS prediction. 

MATE-Net performed favorable prognostic performance and possessed interpretability. We proposed a biomarker, i.e.,  Combined SHAP (C-SHAP) to help clinical prognosis assessment. 

## Prerequisites
The implementation was based on Python 3.8 and the following dependencies: 
1. matplotlib==3.5.1 
2. medcam==0.1.21 
3. nibabel==3.2.2 
4. numpy==1.22.3 
5. pandas==1.4.3 
6. scikit_learn==1.0.2 
7. shap==0.39.0 
8. torch==1.13.1 
9. torchio==0.18.30 
10. torchmetrics==0.11.4 
11. tqdm==4.65.0 

## *** LATEST UPDATE ***
The code has been updated and optimized for clarity and conciseness, eliminating redundancies. To facilitate testing, inputs and labels are randomly generated using *get_example.py* to simulate the workflow.

## Documentation
*wrapper.py* contains the interfaces for training, evaluation and interpretability analysis. 

The code is designed to run on GPU by default.

### 1 Example Data Generation
```
python get_example.py
```

20 data for training and 5 data for validation

### 2 Attention Module Training
```
python attn_train.py
```

### 3 Prediction Module Training
```
python pred_train.py
```

### 4 Prediction Module Validation
```
python pred_validate.py
```

### 5 SHAP Interpretion and C-SHAP generation
```
python pred_interprete.py
```


## DEMO
The C-SHAP values for the four individuals in Figure. 5 in the text are presented in the *demo* folder.

