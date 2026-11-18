# European Bank Customer Churn
Reproducible research prototype. Run from this directory with Python 3.13:

```bash
python -m pip install -r requirements.txt
python train.py /path/to/European_Bank.csv
streamlit run app.py
```

Training is optional for the included fitted model. The original input is not bundled because it contains names and IDs. Keep it separate and restrict access. Only load trusted joblib artifacts. Results include model comparison at 0.50, selected-threshold metrics, EDA, subgroup diagnostics and explanation plots. `features.py` owns the shared scoring transformation. `probabilities.csv` contains anonymous held-out scores only. `shap_values.csv` contains approximate probability-space Shapley contributions for 80 held-out examples, with no identifiers. Streamlit is local, not publicly hosted.
