from pathlib import Path
import json,joblib,pandas as pd,streamlit as st
P=Path(__file__).resolve().parent/'results'
st.set_page_config(page_title='Bank Churn Lab',layout='wide')
@st.cache_resource
def load():return joblib.load(P/'model.joblib')
s=json.loads((P/'summary.json').read_text()); model=load()
st.title('Bank Churn Lab');st.caption('2025 snapshot • Research prototype • Not approved for customer decisions')
a,b,c,d=st.tabs(['Risk calculator','Score distribution','Model explanations','Validation'])
with a:
    left,right=st.columns(2)
    with left:
        x=dict(Geography=st.selectbox('Geography',['France','Spain','Germany']),Gender=st.selectbox('Gender (dataset categories)',['Female','Male']),Age=st.number_input('Age',18,100,42),Tenure=st.number_input('Tenure (years)',0,10,3),CreditScore=st.number_input('Credit score',300,900,650),Balance=st.number_input('Balance',0.,500000.,75000.),EstimatedSalary=st.number_input('Estimated annual salary',0.,500000.,75000.),HasCrCard=int(st.checkbox('Has credit card',True)),IsActiveMember=int(st.checkbox('Active member',False)),NumOfProducts=st.slider('Number of products',1,4,1))
    with right:
        p=model.predict_proba(pd.DataFrame([x]))[0,1];st.metric('Estimated churn probability',f'{p:.1%}');st.write(f"Flag: **{'Review' if p>=s['threshold'] else 'No flag'}** · threshold {s['threshold']:.0%}")
        st.subheader('What-if scenario'); xx=x|dict(IsActiveMember=int(st.checkbox('Scenario: active member',True)),NumOfProducts=st.slider('Scenario: number of products',1,4,2));q=model.predict_proba(pd.DataFrame([xx]))[0,1];st.metric('Scenario probability',f'{q:.1%}',f'{100*(q-p):+.1f} pp',delta_color='off');st.caption('A model sensitivity check, not an estimate of intervention impact.')
with b: st.image(str(P/'distribution.png'));st.image(str(P/'calibration.png'))
with c:
    for k in ['importance','shap','partial_dependence']:st.image(str(P/f'{k}.png'))
    st.caption('These are descriptive model explanations, not causal effects. Hypothetical feature combinations may be uncommon.')
with d:
    st.write(s['model']);st.dataframe(pd.DataFrame([s['test']],index=['Selected threshold']));st.dataframe(pd.read_csv(P/'models.csv'));st.dataframe(pd.read_csv(P/'subgroups.csv'));st.image(str(P/'roc.png'))
