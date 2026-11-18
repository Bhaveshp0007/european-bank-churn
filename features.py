import pandas as pd
CAT=['Geography','Gender']
NUM=['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
RAW=CAT+NUM
def engineer(x):
    x=x[RAW].copy()
    x['BalanceToSalary']=x.Balance/x.EstimatedSalary.clip(lower=1)
    x['ProductDensity']=x.NumOfProducts/(1+x.Tenure)
    x['EngagementProducts']=x.IsActiveMember*x.NumOfProducts
    x['AgeTenure']=x.Age*x.Tenure
    return x
