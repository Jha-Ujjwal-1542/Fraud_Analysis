import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
from DataUtils.datautils import *
warnings.filterwarnings("ignore")
from tqdm import tqdm
import pickle
import logging

#pickle : module for serializing and deserializing Python objects
# tqdm is a popular library that provides a progress bar interface for iterating over sequences(loops), 
# such as lists, tuples, and iterators.

# Logging is a useful technique in software development for recording information
# about the program's execution, including informational messages, warnings, errors, and other relevant details
logging.basicConfig(filename='logname',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger('urbanGUI')


"""load data files"""
target_data = pd.read_csv('data/Train-1542865627584.csv')
tr_data_beneficiary = pd.read_csv('data/Train_Beneficiarydata-1542865627584.csv')
tr_data_inpatient = pd.read_csv('data/Train_Inpatientdata-1542865627584.csv')
tr_data_outpatient = pd.read_csv('data/Train_Outpatientdata-1542865627584.csv')





print("target_data : ",target_data.shape)
print("train_data_beneficiary : ",tr_data_beneficiary.shape)
print("train_data_inpatient : ",tr_data_inpatient.shape)
print("tr_data_outpatient : ",tr_data_outpatient.shape)


logging.info("target_data : ",target_data.shape)
#logging.info("target_data shape: %s", target_data.shape)
logging.info("train_data_beneficiary : ",tr_data_beneficiary.shape)
logging.info("train_data_inpatient : ",tr_data_inpatient.shape)
logging.info("tr_data_outpatient : ",tr_data_outpatient.shape)


#check all the columns of the Inpateint dataset and top 5 data points
print('Columns in this dataset are : ')
print(tr_data_inpatient.columns)
logging.info('Columns in this dataset are : ')
logging.info(tr_data_inpatient.columns)



#create a dataframe for concatinating all Procedure Code columns
inpatient_df = pd.DataFrame(columns = ['ProcedureCode'])
inpatient_df['ProcedureCode'] = pd.concat([tr_data_inpatient["ClmProcedureCode_1"],
                                           tr_data_inpatient["ClmProcedureCode_2"],
                                           tr_data_inpatient["ClmProcedureCode_3"],
                                           tr_data_inpatient["ClmProcedureCode_4"],
                                           tr_data_inpatient["ClmProcedureCode_5"],
                                           tr_data_inpatient["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
print(inpatient_df.shape)
logging.info(inpatient_df.shape)

analyse_cat_columns(inpatient_df,col_to_analyse='ProcedureCode' , prefix = 'Procedure_',
                    title = 'Procedure Distribution- Inpatient(In percentage)',top_val = 30)



#create a dataframe for concatinating all DiagnosisCode Code columns
inpatient_df = pd.DataFrame(columns = ['DiagnosisCode'])
inpatient_df['DiagnosisCode'] = pd.concat([tr_data_inpatient["ClmDiagnosisCode_1"],
                                           tr_data_inpatient["ClmDiagnosisCode_2"],
                                           tr_data_inpatient["ClmDiagnosisCode_3"],
                                           tr_data_inpatient["ClmDiagnosisCode_4"],
                                           tr_data_inpatient["ClmDiagnosisCode_5"],
                                           tr_data_inpatient["ClmDiagnosisCode_6"],
                                           tr_data_inpatient["ClmDiagnosisCode_7"],
                                           tr_data_inpatient["ClmDiagnosisCode_8"],
                                           tr_data_inpatient["ClmDiagnosisCode_9"],
                                           tr_data_inpatient["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()

print(inpatient_df.shape)
logging.info(inpatient_df.shape)

analyse_cat_columns(inpatient_df,col_to_analyse='DiagnosisCode' ,
                    prefix = 'Diagnosis_',
                    title = 'Diagnosis Distribution- Inpatient(In percentage)',
                    top_val = 30, color ='blue')




#check all the columns of the Outpateint dataset and top 5 data points
print('Columns in this dataset are : ')
print(tr_data_outpatient.columns)
logging.info('Columns in this dataset are : ')
logging.info(tr_data_outpatient.columns)


#create a dataframe for concatinating all Procedure Code columns
outpatient_df = pd.DataFrame(columns = ['ProcedureCode'])
outpatient_df['ProcedureCode'] = pd.concat([tr_data_outpatient["ClmProcedureCode_1"],
                                           tr_data_outpatient["ClmProcedureCode_2"],
                                           tr_data_outpatient["ClmProcedureCode_3"],
                                           tr_data_outpatient["ClmProcedureCode_4"],
                                           tr_data_outpatient["ClmProcedureCode_5"],
                                           tr_data_outpatient["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
print(outpatient_df.shape)
logging.info(outpatient_df.shape)

analyse_cat_columns(outpatient_df,col_to_analyse='ProcedureCode' ,
                    prefix = 'Procedure_',title = 'Procedure Distribution- Outpatient(In percentage)',
                    top_val = 30,color ='orange')



#create a dataframe for concatinating all DiagnosisCode Code columns
outpatient_df = pd.DataFrame(columns = ['DiagnosisCode'])
outpatient_df['DiagnosisCode'] = pd.concat([tr_data_outpatient["ClmDiagnosisCode_1"],
                                           tr_data_outpatient["ClmDiagnosisCode_2"],
                                           tr_data_outpatient["ClmDiagnosisCode_3"],
                                           tr_data_outpatient["ClmDiagnosisCode_4"],
                                           tr_data_outpatient["ClmDiagnosisCode_5"],
                                           tr_data_outpatient["ClmDiagnosisCode_6"],
                                           tr_data_outpatient["ClmDiagnosisCode_7"],
                                           tr_data_outpatient["ClmDiagnosisCode_8"],
                                           tr_data_outpatient["ClmDiagnosisCode_9"],
                                           tr_data_outpatient["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
print(outpatient_df.shape)
logging.info(outpatient_df.shape)

analyse_cat_columns(outpatient_df,col_to_analyse='DiagnosisCode' ,
                    prefix = 'Procedure_',title = 'Diagnosis Distribution- Outpatient(In percentage)',
                    top_val = 30,color ='orange',
                    y_lim = np.arange(0,6))

analyse_date_columns(tr_data_inpatient,'ClaimStartDt', 'ClaimEndDt')
analyse_date_columns(tr_data_inpatient,'DischargeDt', 'DischargeDt')
analyse_date_columns(tr_data_outpatient,'ClaimStartDt', 'ClaimEndDt', palette='viridis')
plt.figure(figsize=(15, 5))



tr_data_inpatient.AttendingPhysician.value_counts().head(30).plot( x=tr_data_inpatient.AttendingPhysician , kind = 'barh')
analyse_cat_columns(tr_data_inpatient,
                    col_to_analyse='AttendingPhysician' ,
                    prefix = '',y_lim = np.arange(0,1,0.1),
                    title = 'AttendingPhysician Distribution- Inpatient(In percentage)',top_val = 50)

plt.figure(figsize=(15, 5))



tr_data_inpatient.OperatingPhysician.value_counts().head(30).plot( x=tr_data_inpatient.OperatingPhysician , kind = 'barh')

analyse_cat_columns(tr_data_inpatient,
                    col_to_analyse='OperatingPhysician' ,
                    prefix = '',y_lim = np.arange(0,1,0.1),
                    title = 'OperatingPhysician Distribution- Inpatient(In percentage)',top_val = 50)

plt.figure(figsize=(15, 5))




tr_data_outpatient.AttendingPhysician.value_counts().head(30).plot( x=tr_data_outpatient.AttendingPhysician , kind = 'barh' , color = 'orange')

analyse_cat_columns(tr_data_outpatient,
                    col_to_analyse='AttendingPhysician' ,
                    prefix = '',y_lim = np.arange(0,0.5,0.1),
                    title = 'AttendingPhysician Distribution- Outpatinet(In percentage)',top_val = 50,
                    color='orange')

plt.figure(figsize=(15, 5))




tr_data_outpatient.OperatingPhysician.value_counts().head(30).plot( x=tr_data_outpatient.OperatingPhysician , kind = 'barh' , color = 'orange')

analyse_cat_columns(tr_data_outpatient,
                    col_to_analyse='OperatingPhysician' ,
                    prefix = '',y_lim = np.arange(0,0.5,0.1),
                    title = 'OperatingPhysician Distribution- Outpatinet(In percentage)',top_val = 50,
                    color='orange')
plt.figure(figsize=(8, 5))
sns.distplot(tr_data_inpatient.InscClaimAmtReimbursed)

ax = tr_data_outpatient['InscClaimAmtReimbursed'].plot.hist(bins=100,range=[0,5000], alpha=0.5, figsize=(8, 6), facecolor='c', edgecolor='k')

val = np.percentile(tr_data_outpatient.InscClaimAmtReimbursed,99.9)




#check all the columns of the  labeled  dataset and top 5 data points
print('Columns in this dataset are : ')
print(target_data.columns)

logging.info('Columns in this dataset are : ')
logging.info(target_data.columns)


# sns.countplot('PotentialFraud',data=target_data['PotentialFraud'])
only_fraud_provider = target_data.loc[target_data['PotentialFraud']=='Yes']
print("Dataset shape : ", only_fraud_provider.shape)
logging.info("Dataset shape : ", only_fraud_provider.shape)


fraud_provider_inpatient_df = pd.merge(tr_data_inpatient, only_fraud_provider, how='inner', on='Provider')
print("Dataset shape : ", fraud_provider_inpatient_df.shape)
logging.info("Dataset shape : ", fraud_provider_inpatient_df.shape)
fraud_provider_inpatient_df.head(5)
print("Dataset shape : ", fraud_provider_inpatient_df.shape)
print("Percentage of fraud cases related to inpatinet data : ",(fraud_provider_inpatient_df.shape[0]/tr_data_inpatient.shape[0])*100)
logging.info("Dataset shape : ", fraud_provider_inpatient_df.shape)
logging.info("Percentage of fraud cases related to inpatinet data : ",(fraud_provider_inpatient_df.shape[0]/tr_data_inpatient.shape[0])*100)
fraud_provider_outpatient_df = pd.merge(tr_data_outpatient, only_fraud_provider, how='inner', on='Provider')
print("Dataset shape : ", fraud_provider_outpatient_df.shape)
logging.info("Dataset shape : ", fraud_provider_outpatient_df.shape)
fraud_provider_outpatient_df.head(5)

print("Dataset shape : ", fraud_provider_outpatient_df.shape)
print("Percentage of fraud cases related to outpatinet data : ",(fraud_provider_outpatient_df.shape[0]/tr_data_outpatient.shape[0])*100)

logging.info("Dataset shape : ", fraud_provider_outpatient_df.shape)
logging.info("Percentage of fraud cases related to outpatinet data : ",(fraud_provider_outpatient_df.shape[0]/tr_data_outpatient.shape[0])*100)




#create a dataframe for concatinating all Procedure Code columns
inpatient_df = pd.DataFrame(columns = ['ProcedureCode'])
inpatient_df['ProcedureCode'] = pd.concat([fraud_provider_inpatient_df["ClmProcedureCode_1"],
                                           fraud_provider_inpatient_df["ClmProcedureCode_2"],
                                           fraud_provider_inpatient_df["ClmProcedureCode_3"],
                                           fraud_provider_inpatient_df["ClmProcedureCode_4"],
                                           fraud_provider_inpatient_df["ClmProcedureCode_5"],
                                           fraud_provider_inpatient_df["ClmProcedureCode_6"]], axis=0, sort=True).dropna()

print(inpatient_df.shape)
logging.info(inpatient_df.shape)
analyse_cat_columns(inpatient_df,col_to_analyse='ProcedureCode' , prefix = 'Procedure_',title = 'Procedure Distribution- Inpatient(In percentage)',top_val = 20)



#create a dataframe for concatinating all DiagnosisCode Code columns
inpatient_df = pd.DataFrame(columns = ['DiagnosisCode'])
inpatient_df['DiagnosisCode'] = pd.concat([fraud_provider_inpatient_df["ClmDiagnosisCode_1"],
                                           fraud_provider_inpatient_df["ClmDiagnosisCode_2"],
                                           fraud_provider_inpatient_df["ClmDiagnosisCode_3"],
                                           fraud_provider_inpatient_df["ClmDiagnosisCode_4"],
                                           fraud_provider_inpatient_df["ClmDiagnosisCode_5"],
                                           fraud_provider_inpatient_df["ClmDiagnosisCode_6"],
                                           fraud_provider_inpatient_df["ClmDiagnosisCode_7"],
                                           fraud_provider_inpatient_df["ClmDiagnosisCode_8"],
                                           fraud_provider_inpatient_df["ClmDiagnosisCode_9"],
                                           fraud_provider_inpatient_df["ClmDiagnosisCode_10"]],
                                          axis=0, sort=True).dropna()

print(inpatient_df.shape)
logging.info(inpatient_df.shape)
analyse_cat_columns(inpatient_df,col_to_analyse='DiagnosisCode' , prefix = 'Diagnosis_',title = 'Diagnosis Distribution- Inpatient(In percentage)',top_val = 20 ,color = 'g')



#create a dataframe for concatinating all Procedure Code columns
inpatient_df = pd.DataFrame(columns = ['ProcedureCode'])
inpatient_df['ProcedureCode'] = pd.concat([fraud_provider_outpatient_df["ClmProcedureCode_1"],
                                           fraud_provider_outpatient_df["ClmProcedureCode_2"],
                                           fraud_provider_outpatient_df["ClmProcedureCode_3"],
                                           fraud_provider_outpatient_df["ClmProcedureCode_4"],
                                           fraud_provider_inpatient_df["ClmProcedureCode_5"],
                                           fraud_provider_outpatient_df["ClmProcedureCode_6"]], axis=0, sort=True).dropna()

print(inpatient_df.shape)
logging.info(inpatient_df.shape)
analyse_cat_columns(inpatient_df,col_to_analyse='ProcedureCode' , prefix = 'Procedure_', title = 'Procedure Distribution- Inpatient(In percentage)',top_val = 20)



#create a dataframe for concatinating all DiagnosisCode Code columns
inpatient_df = pd.DataFrame(columns = ['DiagnosisCode'])
inpatient_df['DiagnosisCode'] = pd.concat([fraud_provider_outpatient_df["ClmDiagnosisCode_1"],
                                           fraud_provider_outpatient_df["ClmDiagnosisCode_2"],
                                           fraud_provider_outpatient_df["ClmDiagnosisCode_3"],
                                           fraud_provider_outpatient_df["ClmDiagnosisCode_4"],
                                           fraud_provider_outpatient_df["ClmDiagnosisCode_5"],
                                           fraud_provider_outpatient_df["ClmDiagnosisCode_6"],
                                           fraud_provider_outpatient_df["ClmDiagnosisCode_7"],
                                           fraud_provider_outpatient_df["ClmDiagnosisCode_8"],
                                           fraud_provider_outpatient_df["ClmDiagnosisCode_9"],
                                           fraud_provider_outpatient_df["ClmDiagnosisCode_10"]],
                                          axis=0, sort=True).dropna()

print(inpatient_df.shape)
logging.info(inpatient_df.shape)
analyse_cat_columns(inpatient_df,col_to_analyse='DiagnosisCode' , prefix = 'Diagnosis_',title = 'Diagnosis Distribution- Inpatient(In percentage)',top_val = 20 ,color = 'g')




#check all the columns of the beneficiary dataset and top 5 data points
print('Columns in this dataset are : ')
print(tr_data_beneficiary.columns)
logging.info('Columns in this dataset are : ')
logging.info(tr_data_beneficiary.columns)

tr_bene_inpat_df = pd.merge(tr_data_beneficiary, fraud_provider_inpatient_df, how='inner', on='BeneID')
print(tr_bene_inpat_df.shape)
logging.info(tr_bene_inpat_df.shape)
tr_bene_inpat_df.head(2)



#tr_data_final.State
plt.figure(figsize=(20, 10))
sns.countplot(x='State',y=None, data=tr_bene_inpat_df, orient ="v", order = tr_bene_inpat_df['State'].value_counts().index)




#tr_data_final.State
plt.figure(figsize=(20, 10))
sns.countplot(x='County',y=None, data=tr_bene_inpat_df, orient ="v", order = tr_bene_inpat_df['County'].value_counts().head(30).index)

tr_bene_outpat_df = pd.merge(tr_data_beneficiary, fraud_provider_outpatient_df, how='inner', on='BeneID')
print(tr_bene_outpat_df.shape)
logging.info(tr_bene_outpat_df.shape)
tr_bene_outpat_df.head(2)



#tr_data_final.State
plt.figure(figsize=(20, 10))
sns.countplot(x='State',y=None, data=tr_bene_outpat_df, orient ="v", order = tr_bene_outpat_df['State'].value_counts().index)



#tr_data_final.State
plt.figure(figsize=(20, 10))
sns.countplot(x='County',y=None, data=tr_bene_outpat_df, orient ="v", order = tr_bene_outpat_df['County'].value_counts().head(30).index)
plt.figure(figsize=(8, 5))
sns.distplot(tr_bene_inpat_df.DOB.apply(get_year))

plt.figure(figsize=(8, 5))
sns.distplot(tr_bene_outpat_df.DOB.apply(get_year))



#merging Inpatint and labeled fradulent providers
inpat_labeled_provider_df = pd.merge(tr_data_inpatient , target_data , how='inner', on='Provider')
print(inpat_labeled_provider_df.shape)
logging.info(inpat_labeled_provider_df.shape)
sns.FacetGrid(inpat_labeled_provider_df, col='PotentialFraud',height=5).map(sns.distplot, "InscClaimAmtReimbursed",).add_legend()
plt.show()
plt.tight_layout()



#calculatinng total money lost ib fradulent encounters
Total_money_lost = inpat_labeled_provider_df.loc[inpat_labeled_provider_df['PotentialFraud']== 'Yes']\
    .InscClaimAmtReimbursed.sum()
print("Total money lost : ",Total_money_lost)
logging.info("Total money lost : ",Total_money_lost)



#merging Outpatint and labeled fradulent providers
outpat_labeled_provider_df = pd.merge(tr_data_outpatient , target_data , how='inner', on='Provider')
print(outpat_labeled_provider_df.shape)
logging.info(outpat_labeled_provider_df.shape)



#calculatinng total money lost ib fradulent encounters
Total_money_lost = outpat_labeled_provider_df.loc[outpat_labeled_provider_df['PotentialFraud']== 'Yes']\
    .InscClaimAmtReimbursed.sum()
print("Total money lost : ",Total_money_lost)
print('Total monney lost as per the data for 2019 = ','241288510+54392610, That is around 290 Million' )

logging.info("Total money lost : ",Total_money_lost)
logging.info('Total monney lost as per the data for 2019 = ','241288510+54392610, That is around 290 Million' )

tr_data_inpatient['is_admitted'] = 1
tr_data_outpatient['is_admitted'] = 0




# Merge in_pt, out_pt and ben df into a single patient dataset
tr_data1 = pd.merge(tr_data_inpatient, tr_data_outpatient,
                    left_on = [ idx for idx in tr_data_outpatient.columns if idx in tr_data_inpatient.columns],
                    right_on = [ idx for idx in tr_data_outpatient.columns if idx in tr_data_inpatient.columns],
                    how = 'outer').\
          merge(tr_data_beneficiary,left_on='BeneID',right_on='BeneID',how='inner')



# Replace values with a binary annotation
tr_data1 = tr_data1.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                   'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2,
                   'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2,
                   'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2, 'Gender': 2 },
                  0)




# Replace values with a binary annotation
tr_data1 = tr_data1.replace({'RenalDiseaseIndicator': 'Y'}, 1).astype({'RenalDiseaseIndicator': 'int64'})#array([0, 1], dtype=int64)



#merging the dataset created in above step with target_data
tr_data_final = pd.merge(tr_data1, target_data , how = 'outer', on = 'Provider' )



# adding extra column target, having binary annotation
tr_data_final['target'] = tr_data_final['PotentialFraud']



# representing potential fraud 1 and 0 representing non potential fraud
tr_data_final['target'] = tr_data_final['target'].map({"Yes":1,"No":0})
tr_data_final['target'] = tr_data_final['target'].astype('category',copy=False)
print(tr_data_final.PotentialFraud.value_counts())
logging.info(tr_data_final.PotentialFraud.value_counts())
sns.countplot(x='PotentialFraud', data=tr_data_final)
print(tr_data_final.shape)
logging.info(tr_data_final.shape)
tr_data_final.head(5)



#check colums for NaN values
tr_data_final.isna().sum()



#data Normalization
#Adding a column is_dead  = 1  DOD is given else is_dead = 0
tr_data_final.loc[tr_data_final['DOD'].isnull(), 'Is_Dead'] = '0'
tr_data_final.loc[(tr_data_final['DOD'].notnull()), 'Is_Dead'] = '1'
tr_data_final['DOB'] =  pd.to_datetime(tr_data_final['DOB'], format='%Y-%m-%d')
tr_data_final['ClaimStartDt'] = pd.to_datetime(tr_data_final['ClaimStartDt'], format='%Y-%m-%d')
tr_data_final['DOB'] = tr_data_final['DOB'].where(tr_data_final['DOB'] < tr_data_final['ClaimStartDt'])
tr_data_final['age'] = (tr_data_final['ClaimStartDt'] - tr_data_final['DOB']).astype('<m8[Y]')


#plotting age
sns.FacetGrid(tr_data_final, hue="target" , size=6, palette='viridis').map(sns.distplot, "age",).add_legend()
plt.title('Histogram for potential fraud based on age')
plt.show()
plt.tight_layout()
ax = sns.countplot(x='Race',data=tr_data_final,hue='target')
ax.set_title("Plot to Analyse relation between Race and fraud/not fraud")
tr_data_final = tr_data_final.drop(['DOD'], axis = 1)
tr_data_final = tr_data_final.drop(['DOB'], axis = 1)
tr_data_final['AdmissionDt'] = pd.to_datetime(tr_data_final['AdmissionDt'] , format = '%Y-%m-%d')
tr_data_final['DischargeDt'] = pd.to_datetime(tr_data_final['DischargeDt'],format = '%Y-%m-%d')
tr_data_final['Num_admit_days'] = ((tr_data_final['DischargeDt'] - tr_data_final['AdmissionDt']).dt.days)+1
tr_data_final.loc[tr_data_final['is_admitted'] == 0, 'Num_admit_days'] = '0'
tr_data_final = tr_data_final.drop(['DischargeDt'], axis = 1)
tr_data_final = tr_data_final.drop(['AdmissionDt'], axis = 1)

#repalcing NaN value with 0
tr_data_final.loc[tr_data_final['DeductibleAmtPaid'].isnull(),'DeductibleAmtPaid'] = 0
potential_fraud_df =tr_data_final.loc[tr_data_final['PotentialFraud'] == 'Yes']
df1 = pd.DataFrame(columns = ['ProcedureCode'])
df1['ProcedureCode'] = pd.concat([potential_fraud_df["ClmProcedureCode_1"],
                                           potential_fraud_df["ClmProcedureCode_2"],
                                           potential_fraud_df["ClmProcedureCode_3"],
                                           potential_fraud_df["ClmProcedureCode_4"],
                                           potential_fraud_df["ClmProcedureCode_5"],
                                           potential_fraud_df["ClmProcedureCode_6"]],
                                 axis=0, sort=True).dropna()

df2 = pd.DataFrame(columns = ['DiagnosisCode'])
df2['DiagnosisCode'] = pd.concat([potential_fraud_df["ClmDiagnosisCode_1"],
                                           potential_fraud_df["ClmDiagnosisCode_2"],
                                           potential_fraud_df["ClmDiagnosisCode_3"],
                                           potential_fraud_df["ClmDiagnosisCode_4"],
                                           potential_fraud_df["ClmDiagnosisCode_5"],
                                           potential_fraud_df["ClmDiagnosisCode_6"],
                                           potential_fraud_df["ClmDiagnosisCode_7"],
                                           potential_fraud_df["ClmDiagnosisCode_8"],
                                           potential_fraud_df["ClmDiagnosisCode_9"],
                                           potential_fraud_df["ClmDiagnosisCode_10"]],
                                 axis=0, sort=True).dropna()

plt.figure(figsize=(15, 5))
plt.subplot(121)
df1.ProcedureCode.value_counts().head(5).plot(kind = 'bar' , title ='Top 5 sucpicious procedure')
plt.subplot(122)
df2.DiagnosisCode.value_counts().head(5).plot(kind = 'bar' , title ='Top 5 sucpicious Diagnosis')
Diag_proce_col = ['ClmAdmitDiagnosisCode', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_10',
                  'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
                  'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
                  'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmProcedureCode_1',
                  'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
                  'ClmProcedureCode_5', 'ClmProcedureCode_6']

tr_data_final[Diag_proce_col]= tr_data_final[Diag_proce_col].replace({np.nan:0})
procedure_col = ['ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4','ClmProcedureCode_5','ClmProcedureCode_6']

diagnosis_col = ['ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4','ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8','ClmDiagnosisCode_9','ClmDiagnosisCode_10']

#top 5 procedure
# 4019.0, 2724.0, 9904.0, 8154.0, 66.0
tr_data_final['pr_4019'] = encoded_cat(tr_data_final,'4019.0',procedure_col)
tr_data_final['pr_2724'] = encoded_cat(tr_data_final,'2724.0',procedure_col)
tr_data_final['pr_9904'] = encoded_cat(tr_data_final,'9904.0',procedure_col)
tr_data_final['pr_8154'] = encoded_cat(tr_data_final,'8154.0',procedure_col)
tr_data_final['pr_66'] =   encoded_cat(tr_data_final,'66.0',procedure_col)

#top 5 diagnosis
#4019,25000,2724,V5869,42731
tr_data_final['di_4019'] = encoded_cat(tr_data_final,'4019',diagnosis_col)
tr_data_final['di_25000'] = encoded_cat(tr_data_final,'25000',diagnosis_col)
tr_data_final['di_2724'] = encoded_cat(tr_data_final,'2724',diagnosis_col)
tr_data_final['di_V5869'] = encoded_cat(tr_data_final,'V5869',diagnosis_col)
tr_data_final['di_42731'] =   encoded_cat(tr_data_final,'42731',diagnosis_col)

for i in Diag_proce_col:
    tr_data_final[i][tr_data_final[i]!=0]= 1

tr_data_final[Diag_proce_col].head(5)

tr_data_final[Diag_proce_col]= tr_data_final[Diag_proce_col].astype(float)

#adding column total_num_diag
tr_data_final['total_num_diag'] = 0
for col in diagnosis_col :
    tr_data_final['total_num_diag']  = tr_data_final['total_num_diag'] + tr_data_final[col]

#adding column total_num_proce
tr_data_final['total_num_proce'] = 0
for col in procedure_col :
    tr_data_final['total_num_proce']  = tr_data_final['total_num_proce'] + tr_data_final[col]

tr_data_final['total_num_diag'] =  tr_data_final['total_num_diag'].astype(float)
tr_data_final['total_num_proce'] =  tr_data_final['total_num_proce'].astype(float)

#Checking we how many distinct BeneID we have in our data
len(list(tr_data_final['BeneID'].unique()))

#Checking we how many distinct BeneID we have in our data
len(list(tr_data_final['ClaimID'].unique()))
val_counts_ = tr_data_final['BeneID'].value_counts()
tr_data_final_beneid_df = val_counts_.to_frame() #store this information in dataframe
tr_data_final_beneid_df.columns = ['count']
tr_data_final_beneid_df['BeneID'] = tr_data_final_beneid_df.index
tr_data_final_beneid_df
total_amt_list = []
mean_amt_list = []

for benid in tqdm(tr_data_final_beneid_df['BeneID']):
    total_amt = tr_data_final.loc[tr_data_final['BeneID'] == benid,'InscClaimAmtReimbursed'].sum()
    mean_amt = total_amt / (tr_data_final_beneid_df.loc[tr_data_final_beneid_df['BeneID'] == benid,'count'])
    total_amt_list.append(total_amt)
    mean_amt_list.append(mean_amt)

mean_list = []
for item in tqdm(mean_amt_list):
    mean_list.append(item[0])
tr_data_final_beneid_df['mean_InscClaimAmtReimbursed']=mean_list
tr_data_final_beneid_df['total_InscClaimAmtReimbursed']= total_amt_list
tr_data_final_beneid_df
tr_data_final_beneid_df.to_csv('tr_data_final_beneid_df.csv',index= False)
with open('total_amt_list.pkl', 'wb') as f:
    pickle.dump(total_amt_list, f)
with open('mean_amt_list.pkl', 'wb') as f:
    pickle.dump(mean_amt_list, f)

tr_data_final = pd.merge(tr_data_final, tr_data_final_beneid_df, how='outer', on='BeneID')
print("Dataset shape : ", tr_data_final.shape)
tr_data_final.head(5)
tr_data_final[['mean_InscClaimAmtReimbursed','total_InscClaimAmtReimbursed']]= tr_data_final[
    ['mean_InscClaimAmtReimbursed','total_InscClaimAmtReimbursed']].astype(float)


#number of unique physicians for each patient
tr_data_final['N_unique_Physicians'] = N_unique_values(tr_data_final[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']])



#encoding types of physicians into numeric values
tr_data_final[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']] = \
    np.where(tr_data_final[['AttendingPhysician','OperatingPhysician', 'OtherPhysician']].isnull(), 0, 1)



# number of different physicians who attend a patient
tr_data_final['N_Types_Physicians'] = tr_data_final['AttendingPhysician'] +  tr_data_final[
    'OperatingPhysician'] + tr_data_final['OtherPhysician']



# patients who was attended by only 1 physicians
tr_data_final['Same_Physician'] = tr_data_final.apply(lambda x: 1 if (
        x['N_unique_Physicians'] == 1 and x['N_Types_Physicians'] > 1) else 0,axis=1)



#determine if 1 physician has multi[le role to attend a patient
tr_data_final['Same_Physician2'] = tr_data_final.apply(lambda x: 1 if (
        x['N_unique_Physicians'] == 2 and x['N_Types_Physicians'] > 2) else 0,axis=1)
tr_data_final[['N_unique_Physicians','N_Types_Physicians','Same_Physician','Same_Physician2']].head()



#plot correlation heat map between features
sns.set(context='notebook', style='whitegrid')
plt.figure(figsize=(30, 15))
corr = tr_data_final.corr()
sns.heatmap(corr, annot=False)



#repalve NaN val with 0
tr_data_final['DiagnosisGroupCode']= tr_data_final['DiagnosisGroupCode'].replace({np.nan:0})



#final check to see if a col val is NaN
col_inf_df=pd.DataFrame()
val = tr_data_final.isnull().sum()
col_inf_df['null_col_count']  = val
col_inf_df['column'] = val.index
col_inf_df.loc[col_inf_df['null_col_count'] !=0]
col_inf_df.null_col_count.value_counts()
for col in tr_data_final.columns:
    print(col)


#saving dataframe to csv file to avoid rework
tr_data_final.to_csv("tr_data_final.csv",index= False)
tr_data_final = pd.read_csv("tr_data_final.csv")
print(tr_data_final.shape)
col_to_remove = ['Provider','BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',\
                 'OperatingPhysician', 'OtherPhysician','ClmAdmitDiagnosisCode','NoOfMonths_PartACov',\
                 'NoOfMonths_PartBCov','DiagnosisGroupCode','PotentialFraud']

tr_data_final.drop(columns=col_to_remove, axis=1, inplace=True)
tr_data_final['target']=tr_data_final['target'].astype(int)
tr_data_final.head()
tr_data_final['target'].value_counts()



# X-parameter and y-parameter for model training
y = tr_data_final['target']
X = tr_data_final.drop('target', axis=1)
X = tr_data_final.drop('target', axis=1)