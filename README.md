# 💬 Sentiment Analysis Web App

An interactive **Streamlit** app that tells you whether a piece of text sounds **positive, negative or neutral**, and shows
**which words decided it**. Built with scikit-learn (TF-IDF + Logistic Regression).

**Live demo:** _add your Streamlit Cloud link here_

## Features
- **Example options:** click 😊 Happy, 😞 Angry or 😐 Neutral to fill the box with an example (every click
  gives a different one), then press Analyze to run it
- **Analyze text:** type or paste your own text and get the sentiment with a confidence bar
- **Word highlighting:** words that pushed the result positive are green, negative are red
- **Understands negation:** "not good" and "not bad" are handled using word pairs
- **Batch analysis:** paste many lines or **attach a CSV file** (your other columns are kept), see counts,
  a table and a chart, and download the results
- **Animated background:** a moving gradient whose colour follows the latest result (green positive,
  red negative, amber neutral)
- **History:** every analysis in your session is listed with a chart
- **About the model:** most positive and negative words, accuracy, confusion matrix

## Project structure
```
sentiment-analysis-app/
├── app.py                    # Streamlit app (4 tabs)
├── sentiment_core.py         # Model, prediction and word-explanation code
├── examples.py               # The example texts behind the option buttons
├── train_model.py            # Trains the model -> model/sentiment_model.sav
├── generate_sample_data.py   # Creates demo data -> data/sample_reviews.csv
├── tests/test_sentiment.py   # Automated tests
├── .streamlit/config.toml    # Light theme, so the animated background looks right
├── requirements.txt
├── data/sample_reviews.csv   # Synthetic demo data
└── model/                    # Trained model and metrics are saved here
```

## Run it
```bash
pip install -r requirements.txt
python train_model.py          # trains on the demo data (takes a few seconds)
streamlit run app.py
```
Open http://localhost:8501.

## Use real data (recommended)
The included `sample_reviews.csv` is **synthetic**, built from templates, so its ~100% accuracy is not
meaningful and the model only knows simple phrases. For a model that works on real text:

1. Download a dataset, for example **IMDB Dataset of 50K Movie Reviews** from Kaggle (columns `review`, `sentiment`).
2. Save it in `data/`, for example `data/IMDB.csv`.
3. Train on it:
   ```bash
   python train_model.py --data data/IMDB.csv --text-col review --label-col sentiment
   ```
Labels can be `positive`, `negative` and `neutral` (or `1`/`0` for positive/negative). The sidebar then shows
the real accuracy. IMDB has no neutral reviews, so a model trained only on it never predicts Neutral.
To get a real Neutral class, use a dataset that includes neutral examples (for example a 3-class tweet
sentiment dataset) and label them `neutral`.

## Run the tests
```bash
pip install pytest
pytest -v
```

## How it works
1. **TF-IDF** turns each text into numbers, counting single words and word pairs ("not good").
2. **Logistic Regression** learns which words and pairs signal negative, neutral or positive.
3. The result is three probabilities, and the largest one is the label.
4. To highlight words, the app removes each word in turn and measures how much the score changes.

## Deploy
Push the folder to GitHub (include `model/sentiment_model.sav` and `model/metrics.json`), then deploy on
share.streamlit.io with `app.py` as the main file. Keep the `scikit-learn` version in `requirements.txt`
the same as the one you trained with.

## Limitations
- The included demo data is synthetic, so the model only recognizes simple, template-like phrases.
  Unusual wording, for example flipping a negative word ("I did not hate it"), can be misread.
- **Neutral** is only as good as the neutral examples the model was trained on. If a text is mixed (some
  praise and some complaints), look at the three percentages as well as the label.
- Sarcasm and slang are hard for a simple model.
- It works best on the kind of text it was trained on.

## Tech stack
Python, pandas, NumPy, scikit-learn, Streamlit
