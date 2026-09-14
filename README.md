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

The four files under `models/` are **not included here** — they're the
trained artifacts your training notebook produces. You need to generate them
yourself and commit them to the repo (see below) before the app will fully
work; without them the app still loads, it just shows a warning per missing
artifact and disables that tab.

## 1. Get your trained model files into the repo

Run your training notebook (e.g. `Air_Quality_Forecasting_FIXED.ipynb`) end to
end. It saves:

- `xgboost_aqi_model.pkl`
- `scaler_X.pkl`
- `scaler_y.pkl`
- `cnn_bilstm_model.keras`

Move/copy those four files into a `models/` folder at the root of this repo
— or drop them in the repo root and update the four `*_PATH` constants near
the top of `app.py` to match wherever you put them.

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
