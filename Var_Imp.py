from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

## fitting a random forest classifier to check the variable importance

X = md[list(set(num_var_f)-set(['days_eng_rgt','days_okta_1signin']))]  #independent columns
#print(X[0:4])
scaler = StandardScaler()
X_std = pd.DataFrame(data=scaler.fit_transform(X), columns=X.columns)
del X
#print(X_std[0:4])
y = md['renewal']  
# print(y[0:4])

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_std, y)

# Print the name and gini importance of each feature
var_imp={}
for i,j in zip(X_std.columns, clf.feature_importances_):
    var_imp[i]=j
#pd.DataFrame.from_dict(var_imp, orient='index')


## storing variables importance in data frame
var_imp={}
for i,j in zip(X_std.columns, clf.feature_importances_):
    var_imp[i]=j
var_imp_pd1=pd.DataFrame.from_dict(var_imp, orient='index')
var_imp_pd1.columns=['var_imp']
var_imp_pd1['var_name']=var_imp_pd1.index
var_imp_pd1.sort_values(by=['var_imp'],ascending=False)


## checking important variables using lasso regression
clf = LogisticRegression(penalty="l1",solver='liblinear',)
clf.fit(X_std,y)
thetaLasso=clf.coef_

## storing variables importance in data frame
var_imp={}
for i,j in zip(X_std.columns, clf.coef_.reshape(len(X_std.columns),1)):
    var_imp[i]=j
var_imp_pd=pd.DataFrame.from_dict(var_imp, orient='index')
var_imp_pd.columns=['var_beta']
var_imp_pd['var_beta_abs']=var_imp_pd['var_beta'].abs()
var_imp_pd['var_name']=var_imp_pd.index
var_imp_pd=var_imp_pd.sort_values(by=['var_beta_abs'],ascending=False)


# comparing important variables from Random forest and lasso regression

var_imp_pd=var_imp_pd.sort_values(by=['var_beta_abs'],ascending=False)
var_imp_pd1=var_imp_pd1.sort_values(by=['var_imp'],ascending=False)
var_imp_pd['rank_lasso']=np.arange(len(var_imp_pd))
var_imp_pd1['rank_rfc']=np.arange(len(var_imp_pd1))
print(var_imp_pd.shape)
print(var_imp_pd1.shape)
t=var_imp_pd.merge(var_imp_pd1, on='var_name', how='outer')
t=t.sort_values(by=['var_beta_abs'],ascending=False)
print('variables left out by RFC as unimp but Lasso find imp:')
print(t[(t.rank_lasso<=15) & (t.rank_rfc >15) ])
print('variables left out by Lasso as unimp but RFC find imp:')
print(t[(t.rank_lasso >15) & (t.rank_rfc <=15)]) ##lasso capture usage score to compensate for amount realted vars
t.sort_values(by=['rank_rfc'])

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

##--- information value
# max_bin = 20
# force_bin = 3
# final_iv, IV = data_vars(X_std,X_std.renewal)
# IV=IV.sort_values('IV',ascending=False)
# IV.columns=[x.lower() for x in IV.columns]
# IV['iv_rank']=np.arange(len(IV))
# print(IV)

##--- information value for categorical vars

max_bin = 20
force_bin = 4

X_std=md[cat_var_f]
X_std['renewal']=md['renewal']

final_iv, IV = data_vars(X_std,X_std.renewal)
del X_std

IV=IV.sort_values('IV',ascending=False)
IV.columns=[x.lower() for x in IV.columns]
IV['iv_rank']=np.arange(len(IV))
print(IV)

#---------------------------------------------------------------##
##-------------------- CATEGORICAL VARIABLES -------------------##
## function to perform chi square test on categorical varibales

def chi_sq_func(i,prob=0.95):
    d_ch=md[(md[i].notnull()) & (md.cmp_base_flag=='Yes')]
    contingency_table = pd.crosstab(
        d_ch[d_ch[i].notnull()]['renewal'],
        d_ch[d_ch[i].notnull()][i],
        margins = True
    )
    tab=contingency_table.iloc[0:(contingency_table.shape[0]-1),0:(contingency_table.shape[1]-1)]
    stat, p, dof, expected = chi2_contingency(tab)
    print('dof=%d' % dof)
    critical = chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    print('p-value is: {}'.format(p))
    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    return()

# chi square test of independence for categorical vars
from scipy.stats import chi2_contingency
from scipy.stats import chi2

for i in cat_var:
    print('--- chi square test for column: '+i)
    #if md[md[i].notnull()][i].nunique()>1:
    if md[(md[i].notnull()) & (md.cmp_base_flag=='Yes')][i].nunique()>1:
        chi_sq_func(i)
    else:
        print('not enough variation to perform chi sq test')

## checking distribution of renewal across all categorical var 
for i in cat_var:
    l=md.groupby([i,'renewal'],as_index=False )['cmp_base_flag'].count()
    r=md.groupby([i],as_index=False )['cmp_base_flag'].count()
    r.rename(columns={"cmp_base_flag": "total"}, inplace=True)
    l=l.merge(r, on=i, how='left')
    l['prop']=(l['cmp_base_flag']/l['total'])*100
    print('total observations for varaiable-'+i+' are {}'.format(md[i].count().sum()))
#     print('missing value prop: {}'.format(md[i].isnull().sum()/md.shape[0]))
    print(l[(l.renewal==1) & (l.cmp_base_flag >30) ])


#---------------------------------------------------------------------------##
##-------------------- NUMERICAL + CATEGORICAL VARIABLES -------------------##
## creating a dataset with numerical and dummified categorical variables

X = md[list(set(num_var_f)-set(['days_eng_rgt','days_okta_1signin']))]  #independent columns

scaler = StandardScaler()
X_std = pd.DataFrame(data=scaler.fit_transform(X), columns=X.columns).reset_index(drop=True) 
del X

ct=pd.get_dummies(md[cat_var_f],drop_first=True).reset_index(drop=True) 

X=pd.concat([X_std,ct],axis=1)
print(X.shape)

del X_std
del ct

y = md['renewal']  

# Create a random forest classifier
clf1 = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
# Train the classifier
clf1.fit(X, y)

## storing variable importances in a dataset
var_imp={}
for i,j in zip(list(X.columns), clf1.feature_importances_):
# for i,j in zip(list(X.columns), clf.feature_importances_):
    var_imp[i]=j
var_imp_pd=pd.DataFrame.from_dict(var_imp, orient='index')
var_imp_pd.columns=['var_imp']
var_imp_pd['var_name']=var_imp_pd.index
var_imp_pd=var_imp_pd.sort_values(by=['var_imp'],ascending=False)
var_imp_pd['rank_rfc']=np.arange(len(var_imp_pd))
var_imp_pd[0:25]

y_pred = clf.predict(X)
accuracy_score(y, y_pred)


## fitting lasso regression to identify the important variables
clf = LogisticRegression(penalty="l1",solver='liblinear')

clf.fit(X,y)
thetaLasso=clf.coef_

y_pred = clf.predict(X)
print(accuracy_score(y, y_pred))
confusion_matrix(y, y_pred)

## storing variable importances in a dataset

var_imp1={}
for i,j in zip(X.columns, clf.coef_.reshape(len(X.columns),1)):
    var_imp1[i]=j
var_imp_pd1=pd.DataFrame.from_dict(var_imp1, orient='index')
var_imp_pd1.columns=['var_beta']
var_imp_pd1['var_beta_abs']=var_imp_pd1['var_beta'].abs()
var_imp_pd1['var_name']=var_imp_pd1.index
var_imp_pd1=var_imp_pd1.sort_values(by=['var_beta_abs'],ascending=False)
var_imp_pd1['rank_lasso']=np.arange(len(var_imp_pd1))
