import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import re

import os
#os.getcwd()

### ANOVA ANALYSIS OF THE NUMERICAL VARIABLES
# md is the dataset
anova_res={}
for i in num_var:
    print("{} ========================".format(i))
    df=md[(md[i].notnull())  & (md.cmp_base_flag=='Yes')]
    print('total degree of freedon for anova: {}'.format(df.shape[0]))
    print('degree of freedon for renewal==1 {}'.format(df[df['renewal'] == 1].shape[0]))
    print('degree of freedon for renewal==0: {}'.format(df[df['renewal'] == 0].shape[0]))
    
    o=stats.f_oneway(df[i][df['renewal'] == 0], df[i][df['renewal'] == 1 ])
    print(i + ': the p value for 1way anova: {}'.format(o[1]))
    print(df.groupby(['renewal'])[i].mean())
    l=[o[1]]
#     print(type(l))
    l=l+list(df.groupby(['renewal'])[i].mean())
    l=l+list(df.groupby(['renewal'])[i].median())
    #l=l.append(list(df.groupby(['renewal'])[i].median()))
    print(l)
    anova_res[i]=l

## ANOVA RESULTS STORED IN DATAFRAME
anova_res_pd=pd.DataFrame.from_dict(anova_res,orient='index')
anova_res_pd.columns=['anova_p_value','non_ren_mean','ren_mean','non_ren_md','ren_md']
anova_res_pd['variable_name']=anova_res_pd.index
anova_res_pd
# anova_res_pd.to_csv('1-way anova summary 26may2020.csv')

## funstion to plot num var
## overlaping histograms of the numerical variable for renewing ang not renewing customers
## boxplot of by renewal (dependent varaible) -- normal and log transformed variable
## boxplot of by renewal (dependent varaible) & contract type i.e. lease/rental -- normal and log transformed variable

def plot_func(i,log_add=1):
    print("{} ========================".format(i))
    df=md[(md[i].notnull())  & (md.cmp_base_flag=='Yes')]    

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
    fig.tight_layout()
    
    plt.subplot(1, 2, 1)
    plt.hist(df[i][df.renewal==0],label='non_renewal',alpha=0.5)
    plt.hist(df[i][df.renewal==1],label='rnewal',alpha=0.5,color='red')
    plt.title(i)
    plt.legend(loc='best')
    plt.subplot(1, 2, 2)
    plt.hist(np.log(log_add+df[i][df.renewal==0]),label='non_renewal',alpha=0.5)
    plt.hist(np.log(log_add+df[i][df.renewal==1]),label='renewal',alpha=0.5,color='red')
    plt.title(i+' log transformed')
    plt.legend(loc='best')
    plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
    fig.tight_layout()
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df.renewal,y=df[i])
    plt.title(i)
    plt.legend(loc='best')
    plt.subplot(1, 2, 2)
    sns.boxplot( x=df.renewal,y=np.log(log_add+df[i]))
    plt.title(i+' log transformed')
    plt.legend(loc='best')
    plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    fig.tight_layout()
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df.renewal,y=df[i], hue=df.contract_lro)
    plt.title(i)
    plt.legend(loc='best')
    plt.subplot(1, 2, 2)
    sns.boxplot( x=df.renewal,y=np.log(log_add+df[i]), hue=df.contract_lro)
    plt.title(i+' log transformed')
    plt.legend(loc='best')
    plt.show()

    return()
  
cols=['list_price','trade_in_amount','discount_amount_zdv1', 'list_minus_ti']

corrMatrix = md[cols].corr()
#print (corrMatrix)
sns.heatmap(corrMatrix, annot=True)
plt.show()

print(anova_res_pd[anova_res_pd['variable_name'].isin(cols)])

for i in cols:
    plot_func(i)
    
##----------------- Information Value calculation function
##--------------------------------------------------------

import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

max_bin=20
force_bin = 3

# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)

