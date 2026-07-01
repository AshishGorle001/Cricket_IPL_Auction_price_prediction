# IPL Auction Price Predictor

A Flask web app that predicts a cricket player's IPL auction price using Multiple Linear Regression — trained separately for batsmen, bowlers, and all-rounders, since mixing batting and bowling metrics into one model would blur the signal for each.

## What it does

You either pick an existing player from a dropdown or type in custom stats (matches, runs, strike rate, economy, etc.), and the app returns a predicted auction price in real time. Each player category — Batsman, Bowler, All-Rounder — has its own regression model trained on its own feature set, because "economy rate" means nothing to a batsman's value and "strike rate" means something different for a bowler than a batsman.

On startup, the app also auto-generates 12 visualizations (4 per category) — regression plots, bar charts, and scatter plots — showing how individual stats correlate with price, saved to `static/` and served alongside the predictions.

## Tech stack

| Tool | Role |
|---|---|
| **Flask** | Web framework — routes, forms, templating |
| **pandas** | Loading and querying the player datasets |
| **scikit-learn** | `LinearRegression` — one model per player category |
| **matplotlib + seaborn** | Auto-generated correlation graphs on server start |
| **HTML/Jinja2 templates** | Four pages: home, batsman, bowler, all-rounder |

## Methodology

Built using Ordinary Least Squares (OLS) linear regression, with an 80/20 train/test split on historical auction data. Batsmen models showed an R² of 0.78 — meaning run-scoring metrics are fairly consistent predictors of price — while bowling metrics tend to be more volatile season-to-season, since factors like conditions and match situation swing bowling economy more than they swing a batsman's runs.

## The three models

**Bowlers** — trained on Matches, Wickets, Economy, Average, Strike Rate, Experience
**Batsmen** — trained on Matches, Runs, Average, Strike Rate, Centuries, Experience
**All-Rounders** — trained on both batting stats (Runs, Average_Bat, StrikeRate_Bat, Fifties) and bowling stats (Wickets, Economy, Average_Bowl, StrikeRate_Bowl), plus Matches and Experience — the widest feature set, since all-rounder value comes from two separate skill sets at once

Each is a straightforward `LinearRegression().fit(X, y)` — no regularization or ensembling, kept intentionally simple so the coefficients stay interpretable: you can point to exactly how much one extra wicket or one extra century moves the predicted price.

## How it works — step by step

1. **Load data** — three CSVs (`bowlers_dataset.csv`, `batsmen_dataset.csv`, `allrounders_dataset.csv`), each with ~50 real IPL players and their career stats plus historical auction price
2. **Train three models** — one `LinearRegression` fit per category, all at server startup
3. **Generate graphs** — `generate_all_graphs()` runs once on launch, producing 4 seaborn plots per category (regression line, bar chart, sized scatter plot) saved as PNGs
4. **Serve predictions** — each route (`/batsman`, `/bowler`, `/allrounder`) handles both picking an existing player from the dataset or entering fully custom stats, runs it through the matching model, and returns the predicted price

## Routes

| Route | What it does |
|---|---|
| `/` | Home page |
| `/batsman` | Batsman price prediction — pick a player or enter custom stats |
| `/bowler` | Bowler price prediction |
| `/allrounder` | All-rounder price prediction |

## Running it in Virtual Environment

```bash
python -m venv venv
Venv/Scripts/activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Dataset

Each CSV holds real player names with career-average stats and a target `Price` column (in ₹ Crores/Lakhs), used as the training label:

```
Player,Matches,Runs,Average,StrikeRate,Centuries,Experience,Price
Kohli,220,7000,52,135,5,12,15
Rohit,210,6500,48,140,4,11,14
```

## Team

Built as a group project alongside Y Lakshman, K Sai Ganesh, and T E Santosh.

## Notes on this repo

This started as a two-model (Batsman/Bowler) project — reflected in the original presentation — and was extended to a third All-Rounder model with its own combined batting-and-bowling feature set, plus a full Flask front end for interactive predictions rather than a static analysis.
