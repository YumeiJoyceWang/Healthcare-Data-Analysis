## MIMIC-III clinical data analysis

### Four binary classification tasks:
- Mortality prediction
- Readmission prediction
- Heart failure prediction
- Sepsis prediction

### Dataset description

Source of the data: the MIMIC-III clinical database (https://physionet.org/content/mimiciii/1.4/)

1. Mortality dataset: used the DEATHTIME variable in the admission table to create mortality label. If DEATHTIME was null then our mortality label would be a 0, otherwise 1.

2. Readmission dataset: if the patient has more than one visit, it would be classified as a 1, otherwise 0. 

For disease prediction, the observation window as before the diagnosis of the disease. 

3. Sepsis dataset: Following the latest sepsis 3.0 definition (SOFA >=2), the ICD9 Code for this disease is 995.92. Sepsis label is 1 if the patient presented sepsis, othewise 0.

4. Heart failure dataset: ICD9 code for heart failure started with 428.x. Heart failure label is 1 if the patient had heart failure and 0 if not.

All 4 datasets were split into train, validation and test sets in the ratio of 0.75 : 0.1 : 0.15 respectively.

