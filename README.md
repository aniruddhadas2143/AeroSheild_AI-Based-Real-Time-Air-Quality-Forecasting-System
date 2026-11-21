# AeroShield — Air Quality Forecasting App

A Streamlit app that predicts `CO(GT)` concentrations from the UCI Air Quality
dataset, using an XGBoost regressor for single-point predictions and a
CNN-BiLSTM deep learning model for 24-hour sequence forecasts.

## Repository layout

```
.
├── app.py                   # Streamlit entrypoint
├── requirements.txt         # Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml          # app theme
└── models/
    ├── xgboost_aqi_model.pkl
    ├── scaler_X.pkl
    ├── scaler_y.pkl
    └── cnn_bilstm_model.keras
```

The four files under `models/` **are included** in this package, already
trained — they were produced by running the fixed training pipeline end to
end on the real UCI Air Quality dataset (9,357 hourly readings), not
placeholders. If you retrain on your own data or with different
hyperparameters, just overwrite these four files with your new ones before
pushing.

For reference, the current models' held-out test performance:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge | 2.80 | 2.29 | -3.35 |
| Random Forest | 0.87 | 0.66 | 0.58 |
| **XGBoost** (deployed) | 0.86 | 0.63 | 0.59 |
| CNN-BiLSTM | 1.39 | 1.12 | -0.06 |

Two things worth knowing about these numbers:

- **Ridge's negative R² and CNN-BiLSTM's negative R² both mean "worse than
  just predicting the mean."** For Ridge, that's a real signal: `CO(GT)`
  doesn't relate to the sensor readings linearly, and the tin-oxide/tungsten-
  oxide sensor channels are highly collinear, which a plain linear model
  handles badly. For the CNN-BiLSTM, it means the architecture/hyperparameters
  need more tuning to be useful on this dataset — the pipeline itself is
  correct (chronological split, real RMSE, no leakage), it's just not
  competitive yet.
- **XGBoost is the strongest model here** and is what `app.py`'s
  single-instance prediction tab actually relies on. Treat the CNN-BiLSTM tab
  as a work-in-progress until it's retuned — more epochs, a different
  architecture, or more engineered features (hour-of-day, day-of-week) would
  likely help.

## 1. If you want to retrain the models yourself

Run your training notebook (e.g. `Air_Quality_Forecasting_FIXED.ipynb`) end to
end. It saves:

- `xgboost_aqi_model.pkl`
- `scaler_X.pkl`
- `scaler_y.pkl`
- `cnn_bilstm_model.keras`

Move/copy those four files into the `models/` folder at the root of this repo
(overwriting the ones already there), or drop them elsewhere and update the
four `*_PATH` constants near the top of `app.py` to match.

**If any of these files is over 100 MB**, GitHub will reject a normal `git
push`. Use [Git LFS](https://git-lfs.github.com) instead:
```bash
git lfs install
git lfs track "models/*.keras" "models/*.pkl"
git add .gitattributes
```

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: AeroShield air quality forecasting app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 3. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. Click **"New app"**.
3. Pick your repository, the `main` branch, and set the main file path to
   `app.py`.
4. Open **"Advanced settings"** before deploying and choose a Python version
   (3.11 is a safe, well-supported choice for this stack). This is the
   reliable way to pin the Python version on Community Cloud right now — a
   `runtime.txt` file is sometimes ignored by the platform.
5. Click **Deploy**. The first build can take a few minutes since it's
   installing TensorFlow.

Your app will be live at `https://<your-app-name>.streamlit.app`.

## 4. Updating the app later

Any push to the branch you deployed from redeploys automatically. To change
the Python version afterwards, you have to delete the app and redeploy it
(Streamlit Cloud doesn't let you change Python version in place).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- `requirements.txt` uses `tensorflow-cpu` instead of `tensorflow` — it's a
  much smaller download and installs faster on Community Cloud's free tier,
  which helps avoid build timeouts/memory limits. Swap it for `tensorflow`
  if you deploy somewhere with more resources.
- If you'd rather not commit model binaries to a public repo, you can instead
  have `app.py` download them from a release asset, S3/GCS bucket, or Hugging
  Face Hub at startup — ask if you'd like that version of `load_artifacts()`.
