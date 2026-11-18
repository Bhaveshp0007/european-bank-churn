from pathlib import Path
import sys,json,joblib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_predict
from sklearn.metrics import *
from sklearn.inspection import permutation_importance,PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from features import RAW,CAT,engineer
P=Path(__file__).resolve().parent/'results';P.mkdir(exist_ok=True)
def save(name):
    plt.tight_layout();plt.savefig(P/f'{name}.png',dpi=145);plt.close()
def main():
    d=pd.read_csv(sys.argv[1]); assert set(d.Exited.unique())=={0,1} and d.CustomerId.is_unique
    y=d.Exited; X=d[RAW]; a,b,c,e=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    prep=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),[k for k in engineer(X).columns if k not in CAT]),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),CAT)])
    models={'Logistic regression':LogisticRegression(max_iter=1000),'Decision tree':DecisionTreeClassifier(max_depth=5,min_samples_leaf=30,random_state=42),'Random forest':RandomForestClassifier(n_estimators=220,min_samples_leaf=5,random_state=42,n_jobs=-1),'Gradient boosting':GradientBoostingClassifier(random_state=42)}
    rows=[]; fitted={};oofs={};preds={}
    def metrics(y,p,t):
        z=p>=t;tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
        return dict(accuracy=accuracy_score(y,z),precision=precision_score(y,z,zero_division=0),recall=recall_score(y,z),f1=f1_score(y,z),roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),brier=brier_score_loss(y,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
    for name,model in models.items():
        f=make_pipeline(FunctionTransformer(engineer),prep,model)
        o=cross_val_predict(f,a,c,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')[:,1];oofs[name]=o
        from sklearn.base import clone
        f=clone(f).fit(a,c);fitted[name]=f;p=f.predict_proba(b)[:,1];preds[name]=p
        rows.append(dict(model=name,oof_auc=roc_auc_score(c,o),**metrics(e,p,.5)))
    table=pd.DataFrame(rows).sort_values('oof_auc',ascending=False);table.to_csv(P/'models.csv',index=False)
    name=table.iloc[0].model;f=fitted[name];p=preds[name];o=oofs[name];grid=np.arange(.05,.951,.01);t=float(round(max(grid,key=lambda k:(f1_score(c,o>=k),k)),2))
    m=metrics(e,p,t);base=metrics(e,p,.5)
    joblib.dump(f,P/'model.joblib'); pd.DataFrame({'probability':p}).to_csv(P/'probabilities.csv',index=False)
    profiles=b.copy();profiles['Exited']=e;profiles['Probability']=p;profiles['Flag']=p>=t
    for k in ['Geography','Gender','IsActiveMember','NumOfProducts']:
        d.groupby(k).Exited.agg(['size','mean']).to_csv(P/f'eda_{k}.csv')
    audit=[]
    for k in ['Gender','Geography']:
        for g,s in profiles.groupby(k):
            q=metrics(s.Exited,s.Probability,t);audit.append(dict(attribute=k,group=g,n=len(s),prevalence=s.Exited.mean(),flag_rate=s.Flag.mean(),recall=q['recall'],fpr=q['fp']/(q['fp']+q['tn'])))
    pd.DataFrame(audit).to_csv(P/'subgroups.csv',index=False)
    perm=permutation_importance(f,b,e,n_repeats=6,random_state=42,scoring='roc_auc');imp=pd.Series(perm.importances_mean,index=RAW).sort_values(ascending=False);imp.rename('auc_drop').to_csv(P/'importance.csv');imp.sort_values().plot.barh(title='Held-out permutation importance');save('importance')
    for n, pp in preds.items(): RocCurveDisplay.from_predictions(e,pp,name=n,ax=plt.gca())
    plt.plot([0,1],[0,1],'--',color='gray');save('roc')
    plt.hist([p[e==0],p[e==1]],bins=20,label=['Retained','Churned']);plt.axvline(t,color='black',linestyle='--');plt.xlabel('Churn probability');plt.legend();save('distribution')
    from sklearn.calibration import CalibrationDisplay
    CalibrationDisplay.from_predictions(e,p,n_bins=10,strategy='quantile');save('calibration')
    fig,axes=plt.subplots(1,3,figsize=(11,3))
    for ax,k in zip(axes,['Age','NumOfProducts','IsActiveMember']):
        grid=np.unique(np.quantile(b[k],np.linspace(.05,.95,20)).round()) if k=='Age' else sorted(b[k].unique())
        ax.plot(grid,[f.predict_proba(b.assign(**{k:v}))[:,1].mean() for v in grid],marker='o');ax.set_xlabel(k);ax.set_ylabel('Mean predicted risk')
    save('partial_dependence')
    # Monte Carlo interventional Shapley values in probability space. Each path starts at a sampled training row.
    rng=np.random.default_rng(42); sample=b.sample(80,random_state=42); bg=a.sample(64,random_state=42).reset_index(drop=True); vals=[]
    for _,row in sample.iterrows():
        states=[bg.copy()]; orders=np.array([rng.permutation(len(RAW)) for _ in range(len(bg))]); cur=bg.copy()
        for j in range(len(RAW)):
            for i,k in enumerate(RAW):cur.loc[orders[:,j]==i,k]=row[k]
            states.append(cur.copy())
        pp=f.predict_proba(pd.concat(states,ignore_index=True))[:,1].reshape(len(RAW)+1,-1)
        phi=np.empty_like(orders,dtype=float);phi[np.arange(len(bg))[:,None],orders]=np.diff(pp,axis=0).T
        assert np.isclose(phi.mean(axis=0).sum(),pp[-1].mean()-pp[0].mean())
        vals.append(phi.mean(axis=0))
    sh=pd.DataFrame(vals,columns=RAW);sh.to_csv(P/'shap_values.csv',index=False);sh.abs().mean().sort_values().plot.barh(title='Mean absolute approximate SHAP value');plt.xlabel('Probability units');save('shap')
    (P/'summary.json').write_text(json.dumps(dict(model=name,threshold=t,test=m,default=base,n=len(d),train=len(a),test_n=len(b),churn=float(y.mean()),missing=int(d.isna().sum().sum()),year=d.Year.unique().tolist(),top=imp.index[:3].tolist()),indent=2))
if __name__=='__main__':main()
