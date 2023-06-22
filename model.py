
'''Model for fraud analysis'''

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from Utils.modelUtils import *
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_curve, auc , f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


tr_data_final = pd.read_csv("tr_data_final.csv")
print(tr_data_final.shape)
tr_data_final.head(2)

col_to_remove = ['Provider','BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',\
                 'OperatingPhysician', 'OtherPhysician','ClmAdmitDiagnosisCode','NoOfMonths_PartACov',\
                 'NoOfMonths_PartBCov','DiagnosisGroupCode','PotentialFraud']

tr_data_final.drop(columns=col_to_remove, axis=1, inplace=True)
tr_data_final['target']=tr_data_final['target'].astype(int)

tr_data_final.head()

tr_data_final['target'].value_counts()

y = tr_data_final['target']
X = tr_data_final.drop('target', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("Y_train: ", y_train)
print("Shapes data into train,test,CV splitting..")
print("Training X : ",X_train.shape," | Training Y : ",y_train.shape)
print("Test X : ",X_test.shape," | Test Y : ",y_test.shape)

"""# Handling Numerical Column"""

from sklearn.preprocessing import Normalizer , StandardScaler

def num_col_normalizer(X,col=''):
    """This function retruns normalised column for train and test data"""
    normalizer = Normalizer()

    normalizer.fit(X[col].values.reshape(1,-1))

    tr = normalizer.transform(X[col].values.reshape(1,-1)).reshape(-1,1)


    return tr

#InscClaimAmtReimbursed
tr= num_col_normalizer(X_train,col='InscClaimAmtReimbursed')
te= num_col_normalizer(X_test,col='InscClaimAmtReimbursed')
X_train['InscClaimAmtReimbursed'] = tr
X_test ['InscClaimAmtReimbursed'] = te

tr= num_col_normalizer(X_train,col='DeductibleAmtPaid')
te= num_col_normalizer(X_test,col='DeductibleAmtPaid')
X_train['DeductibleAmtPaid'] = tr
X_test ['DeductibleAmtPaid'] = te

tr= num_col_normalizer(X_train,col='IPAnnualReimbursementAmt')
te= num_col_normalizer(X_test,col='IPAnnualReimbursementAmt')
X_train['IPAnnualReimbursementAmt'] = tr
X_test ['IPAnnualReimbursementAmt'] = te

tr= num_col_normalizer(X_train,col='IPAnnualDeductibleAmt')
te= num_col_normalizer(X_test,col='IPAnnualDeductibleAmt')
X_train['IPAnnualDeductibleAmt'] = tr
X_test ['IPAnnualDeductibleAmt'] = te

tr = num_col_normalizer(X_train,col='OPAnnualReimbursementAmt')
te = num_col_normalizer(X_test,col='OPAnnualReimbursementAmt')
X_train['OPAnnualReimbursementAmt'] = tr
X_test ['OPAnnualReimbursementAmt'] = te

tr= num_col_normalizer(X_train,col='OPAnnualDeductibleAmt')
te= num_col_normalizer(X_test,col='OPAnnualDeductibleAmt')
X_train['OPAnnualDeductibleAmt'] = tr
X_test ['OPAnnualDeductibleAmt'] = te

tr = num_col_normalizer(X_train,col='mean_InscClaimAmtReimbursed')
te= num_col_normalizer(X_test,col='mean_InscClaimAmtReimbursed')
X_train['mean_InscClaimAmtReimbursed'] = tr
X_test ['mean_InscClaimAmtReimbursed'] = te

tr = num_col_normalizer(X_train,col='total_InscClaimAmtReimbursed')
te = num_col_normalizer(X_test,col='total_InscClaimAmtReimbursed')
X_train['total_InscClaimAmtReimbursed'] = tr
X_test ['total_InscClaimAmtReimbursed'] = te


tr= num_col_normalizer(X_train,col='age')
te= num_col_normalizer(X_test,col='age')
X_train['age'] = tr
X_test ['age'] = te

tr = num_col_normalizer(X_train,col='Num_admit_days')
te= num_col_normalizer(X_test,col='Num_admit_days')
X_train['Num_admit_days'] = tr
X_test ['Num_admit_days'] = te

tr = num_col_normalizer(X_train,col='N_unique_Physicians')
te = num_col_normalizer(X_test,col='N_unique_Physicians')
X_train['N_unique_Physicians'] = tr
X_test ['N_unique_Physicians'] = te


tr= num_col_normalizer(X_train,col='N_Types_Physicians')
te= num_col_normalizer(X_test,col='N_Types_Physicians')
X_train['N_Types_Physicians'] = tr
X_test ['N_Types_Physicians'] = te


# for checking the colum of x_train
# print(X_train,y_train)
# for t,k in X_train.items():
#     print(t)


# exit()






'''Model: logistic Regression'''


grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
# logreg=LogisticRegression()
# logreg_cv=GridSearchCV(logreg,grid,cv=10,  n_jobs=-1, return_train_score=True)
# logreg_cv.fit(X_train,y_train)


# print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
# print("accuracy :",logreg_cv.best_score_)

logreg2=LogisticRegression(C=0.01,penalty="l2")
logreg2.fit(X_train,y_train)
print("score",logreg2.score(X_test,y_test))
filename = 'LR_model.sav'
pickle.dump(logreg2, open(filename, 'wb'))


model_f1_score_LR, model_AUC_score_LR = model_performence_check(logreg2,X_train,X_test,y_train,y_test)








'''Model: Random Forest'''

RF = RandomForestClassifier(class_weight = 'balanced', random_state=42)


#paramet tuning 
param_grid = {
    'n_estimators': [300, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
###
# #hyperParameter tuning : cv grid search cv
# RF_CV = GridSearchCV(estimator=RF, param_grid=param_grid, cv= 5,scoring='roc_auc', n_jobs=-1, return_train_score=True, verbose=10)
# RF_CV.fit(X_train, y_train)

# filename = 'RF_model.sav'
# pickle.dump(RF_CV, open(filename, 'wb'))

# print("tuned hpyerparameters :(best parameters) ",RF_CV.best_params_)
# print("roc_auc :",RF_CV.best_score_)

# ##



RF2 = RandomForestClassifier(n_estimators = 300,max_features='auto',max_depth=8,criterion='gini',
                             class_weight = 'balanced',n_jobs=-1,verbose=10, random_state=42)

RF2=RF2.fit(X_train,y_train)

print("score",RF2.score(X_test,y_test))

filename = 'RF_model.sav'
pickle.dump(RF2, open(filename, 'wb'))

model_f1_score_RF, model_AUC_score_RF = model_performence_check(RF2,X_train,X_test,y_train,y_test)


features = tr_data_final.columns
importances = RF2.feature_importances_
indices = (np.argsort(importances))[-30:]
plt.figure(figsize=(8,7))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()




model = ['Logistic Regression','Random Forest']
f1_score = [model_f1_score_LR, model_f1_score_RF]
AUC_score = [model_AUC_score_LR, model_AUC_score_RF]

model_comp(model,f1_score,AUC_score,'Scores','Model performence Summary','f1_score','AUC_score')



