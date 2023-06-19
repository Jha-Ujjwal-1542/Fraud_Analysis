import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def analyse_cat_columns(dataset,col_to_analyse='' ,
                        prefix = '',title = 'Distribution in percentage ',
                        top_val = 30,
                        y_lim = np.arange(0,8),
                        color ='blue'):
    #get top 30 procedure codes with its count
    val_counts_ = dataset[col_to_analyse].value_counts()
    val_counts_df = val_counts_.to_frame() #store this information in dataframe
    val_counts_df.columns = ['count']
    val_counts_df[col_to_analyse] = val_counts_df.index

    #since simply plotting bar plots for count of each val would not give much information,so it better to plot in percentage.
    val_counts_df[col_to_analyse] = prefix + val_counts_df[col_to_analyse].astype(str)
    val_counts_df['Percentage'] = (val_counts_df['count']/sum(val_counts_df['count']))*100
    val_counts_df = val_counts_df.head(top_val)
    val_counts_df.plot(x =col_to_analyse, y='Percentage', kind='bar', color = color, \
                        title=title, figsize=(15,5),yticks = y_lim ,grid = True )


def get_year(date):
    """This function returns year from date"""
    date = str(date)
    return(int(date[:4]))

def get_month(date):
    """This function returns month from date"""
    date = str(date)
    return (int(date[5:7]))

def analyse_date_columns(data_frame, start_date_col, end_date_col , palette='coolwarm'):
    date_df = pd.DataFrame(columns = ['start_month','start_year','end_month','end_year' ])
    date_df['start_month'] = data_frame[start_date_col].apply(get_month)
    date_df['start_year'] = data_frame[start_date_col].apply(get_year)
    date_df['end_month'] = data_frame[end_date_col].apply(get_month)
    date_df['end_year'] = data_frame[end_date_col].apply(get_year)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    data_frame[start_date_col].apply(get_year).value_counts().plot(kind='bar',title='start date')
    plt.subplot(122)
    data_frame[end_date_col].apply(get_year).value_counts().plot(kind='bar', title='end date')
    fig, axes = plt.subplots(nrows=1, ncols=2 , figsize=(15,5))
    sns.countplot(x='start_month',data=date_df,hue='start_year',palette=palette , ax = axes[0])
    axes[0].set_title("Plot to Analyse behaviour of data point acc to month and year")
    sns.countplot(x='end_month',data=date_df,hue='end_year',palette=palette ,  ax = axes[1])
    axes[1].set_title("Plot to Analyse behaviour of data point acc to month and year ")
    fig, axes = plt.subplots(nrows=1, ncols=2 , figsize=(15,5))
    sns.stripplot(x='start_month',y="start_year", data=date_df, orient = 'h' , ax = axes[0])
    axes[0].set_title("Plot to Analyse behaviour of data point acc to month and year ")
    sns.stripplot(x='end_month',y="end_year", data=date_df, orient = 'h' , ax = axes[1])
    axes[1].set_title("Plot to Analyse behaviour of data point acc to month and year ")
    plt.tight_layout()


def encoded_cat(dataset, feature_to_encode='',col_list=[]):
    """This function returns top 5 cat column useful in determining potential fraud"""
    outer_list =[]
    for col in col_list:
        list_1 = list()

        for item in list(dataset[col]):
            if str(item) == str(feature_to_encode):
                list_1.append(1)
            else:
                list_1.append(0)

        outer_list.append(list_1)
    li_sum = np.array([0]*558211)
    for i in range(0,len(outer_list)):
        li1 = np.array(outer_list[i])
        li_sum = li_sum + li1
    return li_sum

def N_unique_values(df):
    """This function finds the unique values in a df row"""
    return np.array([len(set([i for i in x[~pd.isnull(x)]])) for x in df.values])