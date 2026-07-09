# MSME Liquidity Risk Screener

Streamlit app for the DSA 8201 MSME liquidity constraint prediction model
(stacking ensemble: Random Forest + XGBoost + Gradient Boosting -> logistic meta-learner).

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL it prints (usually http://localhost:8501).

## Deploy for free (Streamlit Community Cloud)

1. Create a new GitHub repo and push this folder's contents to it:
   app.py, model.joblib, feature_columns.json, cat_options.json,
   num_defaults.json, columns_meta.json, requirements.txt.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", select the repo, branch, and app.py as the entry point.
4. Deploy. You will get a public URL like
   https://<your-app-name>.streamlit.app -- this is the "[deployment link]"
   to cite in the article's Abstract and Deployment section.

Free tier limits: app sleeps after inactivity and wakes on the next visit
(~30-60 second cold start) -- normal for a coursework demo.

## Alternative free hosts

- Hugging Face Spaces (choose the Streamlit SDK when creating a Space) --
  similar GitHub-based flow, https://huggingface.co/new-space
- Render.com free web service -- more setup (needs a Dockerfile or start
  command) but no sleep-on-inactivity behaviour during active hours.

## Files

- app.py -- Streamlit UI and inference logic
- model.joblib -- trained StackingClassifier (fit on the full cleaned dataset)
- feature_columns.json -- exact one-hot column layout the model expects
- cat_options.json -- valid category values per categorical field (populates dropdowns)
- num_defaults.json -- median fallback values for numeric fields
- columns_meta.json -- categorical vs numeric column lists
- train_final.py -- script that produced model.joblib and the JSON artefacts

## What the model predicts

Given firm characteristics (country, demographics, financial performance,
and financial product access/behaviour), the app returns the probability
that the business currently reports a binding liquidity constraint. It is
a self-report screening signal, not a validated business-failure or
creditworthiness score -- see the "About this model" note in the app itself.
