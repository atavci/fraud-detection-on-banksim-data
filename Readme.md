# Fraud Detection with Machine Learning On Banksim Data

Fraudulent behavior can be seen across many different fields such as e-commerce, healthcare, payment and banking systems. Fraud costs businesses millions of dollars each year. 

Automated detection of fraudulent behavior can be done in various ways including rule based approaches and machine learning.
This repository uses the latter approach for classification of fraudulent transactions. 

For the in-depth analysis you can check out my notebook on Kaggle below or run the ipynb file in your Juptyter environment. 
https://www.kaggle.com/code/turkayavci/fraud-detection-on-bank-payments/notebook

The synthetically generated dataset consists of payments from various customers made in different time periods and with different amounts. If you want more information on the dataset you can refer below to the Dataset title for the dataset link and information and you can find the original paper for the dataset under the "Original Paper" title. 

For those who do not wish to run the script to acquire the results, here is a quick recap of the classification results of the machine learning models used in the script:

!Update Note: These results are without the oversampling technique SMOTE. I have also added a jupyter notebook with more insights and used the SMOTE for balancing the dataset. Overall results looks more better just check the file called Fraud Detection on Bank Payments.ipynb from inside the repo.

<br/>Classification Report for K-Nearest Neighbours (1:fraudulent,0:non-fraudulent) :

|class | precision | recall | f1-score | support|
| ---- | --------- | ------ | -------- | -------|        
|  0   |   1.00    |   1.00 |  1.00    | 176233 |
|  1   |   0.83    |   0.61 |  0.70    |  2160  |
           
Confusion Matrix of K-Nearest Neigbours:
<br/> [175962    271]
<br/> [   845   1315] 



<br/>Classification Report for XGBoost : 

class | precision | recall | f1-score | support|
| ---- | --------- | ------ | -------- | -------|        
|  0   |   1.00    |   1.00 |  1.00    | 176233 |
|  1   |   0.89    |   0.76 |  0.82    |  2160  |
           
           
Confusion Matrix of XGBoost: 
<br/> [176029    204] 
<br/> [   529   1631] 




<br/>Classification Report for Random Forest Classifier : 

class | precision | recall | f1-score | support|
| ---- | --------- | ------ | -------- | -------|        
|  0   |   1.00    |   0.96 |  0.98    | 176233 |
|  1   |   0.24    |   0.98 |  0.82    |  2160  |
           
         
 Confusion Matrix of Random Forest Classifier: 
<br/> [169552   6681]
<br/> [    39   2121]



<br/>Classification Report for Ensembled Models(RandomForest+KNN+XGBoost) : 

class | precision | recall | f1-score | support|
| ---- | --------- | ------ | -------- | -------|        
|  0   |   1.00    |   1.00 |  1.00    | 176233 |
|  1   |   0.73    |   0.81 |  0.77    |  2160  |
           

Confusion Matrix of Ensembled Models: 
<br/> [175604    629]
<br/> [   417   1743]


## Dataset
https://www.kaggle.com/ntnu-testimon/banksim1

## Original paper

Lopez-Rojas, Edgar Alonso ; Axelsson, Stefan Banksim: A bank payments simulator for fraud detection research Inproceedings 26th European Modeling and Simulation Symposium, EMSS 2014, Bordeaux, France, pp. 144–152, Dime University of Genoa, 2014, ISBN: 9788897999324. https://www.researchgate.net/publication/265736405_BankSim_A_Bank_Payment_Simulation_for_Fraud_Detection_Research
