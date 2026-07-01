import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# --- 1. LOAD DATA & TRAIN MODELS ---
bowlers    = pd.read_csv("bowlers_dataset.csv")
batsmen    = pd.read_csv("batsmen_dataset.csv")
allrounders = pd.read_csv("allrounders_dataset.csv")

# Bowler Model
bowler_X     = bowlers[['Matches','Wickets','Economy','Average','StrikeRate','Experience']]
bowler_y     = bowlers['Price']
bowler_model = LinearRegression().fit(bowler_X, bowler_y)

# Batsman Model
batsman_X     = batsmen[['Matches','Runs','Average','StrikeRate','Centuries','Experience']]
batsman_y     = batsmen['Price']
batsman_model = LinearRegression().fit(batsman_X, batsman_y)

# All-Rounder Model
allrounder_X     = allrounders[['Matches','Runs','Average_Bat','StrikeRate_Bat','Fifties',
                                 'Wickets','Economy','Average_Bowl','StrikeRate_Bowl','Experience']]
allrounder_y     = allrounders['Price']
allrounder_model = LinearRegression().fit(allrounder_X, allrounder_y)

os.makedirs("static", exist_ok=True)


# --- 2. GENERATE ALL GRAPHS ---
def generate_all_graphs():
    """Generates and saves all graphs for batsmen, bowlers and all-rounders."""

    # ── BOWLERS (White Theme) ──────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)

    plt.figure(figsize=(8, 5))
    sns.regplot(x='Wickets', y='Price', data=bowlers,
                scatter_kws={'color': '#0d6efd'}, line_kws={'color': '#dc3545'})
    plt.title("Wickets vs Price")
    plt.tight_layout()
    plt.savefig("static/bowler_graph1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.regplot(x='Economy', y='Price', data=bowlers,
                scatter_kws={'color': '#198754'}, line_kws={'color': '#dc3545'})
    plt.title("Economy vs Price")
    plt.tight_layout()
    plt.savefig("static/bowler_graph2.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Experience', y='Price', data=bowlers,
                hue='Experience', palette='viridis', legend=False)
    plt.title("Experience vs Price")
    plt.tight_layout()
    plt.savefig("static/bowler_graph3.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='StrikeRate', y='Price', size='Wickets',
                    sizes=(50, 300), data=bowlers, color='#6f42c1')
    plt.title("Strike Rate vs Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("static/bowler_graph4.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── BATSMEN (Dark Theme) ───────────────────────────────────────
    sns.set_theme(style="darkgrid", font_scale=1.1)

    plt.figure(figsize=(8, 5))
    sns.regplot(x='Runs', y='Price', data=batsmen,
                scatter_kws={'color': '#0d6efd'}, line_kws={'color': '#fd7e14'})
    plt.title("Runs vs Price")
    plt.tight_layout()
    plt.savefig("static/batsman_graph1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.regplot(x='StrikeRate', y='Price', data=batsmen,
                scatter_kws={'color': '#fd7e14'}, line_kws={'color': '#0d6efd'})
    plt.title("Strike Rate vs Price")
    plt.tight_layout()
    plt.savefig("static/batsman_graph2.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Centuries', y='Price', data=batsmen,
                hue='Centuries', palette='flare', legend=False)
    plt.title("Centuries vs Price")
    plt.tight_layout()
    plt.savefig("static/batsman_graph3.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Average', y='Price', size='Runs',
                    sizes=(50, 300), data=batsmen, color='#20c997')
    plt.title("Average vs Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("static/batsman_graph4.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── ALL-ROUNDERS (Teal/Mixed Theme) ───────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Graph 1: Runs vs Price
    plt.figure(figsize=(8, 5))
    sns.regplot(x='Runs', y='Price', data=allrounders,
                scatter_kws={'color': '#0dcaf0'}, line_kws={'color': '#dc3545'})
    plt.title("Runs vs Price (All-Rounders)")
    plt.tight_layout()
    plt.savefig("static/allrounder_graph1.png", dpi=150)
    plt.close()

    # Graph 2: Wickets vs Price
    plt.figure(figsize=(8, 5))
    sns.regplot(x='Wickets', y='Price', data=allrounders,
                scatter_kws={'color': '#fd7e14'}, line_kws={'color': '#0d6efd'})
    plt.title("Wickets vs Price (All-Rounders)")
    plt.tight_layout()
    plt.savefig("static/allrounder_graph2.png", dpi=150)
    plt.close()

    # Graph 3: Economy vs Price
    plt.figure(figsize=(8, 5))
    sns.regplot(x='Economy', y='Price', data=allrounders,
                scatter_kws={'color': '#198754'}, line_kws={'color': '#dc3545'})
    plt.title("Economy vs Price (All-Rounders)")
    plt.tight_layout()
    plt.savefig("static/allrounder_graph3.png", dpi=150)
    plt.close()

    # Graph 4: Batting Strike Rate vs Price, sized by Wickets
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='StrikeRate_Bat', y='Price', size='Wickets',
                    sizes=(50, 300), data=allrounders, color='#6f42c1')
    plt.title("Batting Strike Rate vs Price (All-Rounders)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("static/allrounder_graph4.png", dpi=150, bbox_inches='tight')
    plt.close()


# Run once on server start
generate_all_graphs()


# --- 3. WEB ROUTES ---

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/bowler", methods=["GET", "POST"])
def bowler():
    prediction   = None
    players      = bowlers.to_dict(orient='records')
    selected     = None

    if request.method == "POST":
        if request.form.get("player"):
            p    = bowlers[bowlers['Player'] == request.form["player"]].iloc[0]
            vals = [[p['Matches'], p['Wickets'], p['Economy'],
                     p['Average'], p['StrikeRate'], p['Experience']]]
            selected = p['Player']
        else:
            vals = [[float(request.form.get(x) or 0)
                     for x in ['matches', 'wickets', 'economy', 'average', 'sr', 'exp']]]
            selected = "Custom Player"

        prediction = round(max(0.0, bowler_model.predict(vals)[0]), 2)

    return render_template("bowler.html", players=players,
                           prediction=prediction, selected_player=selected)


@app.route("/batsman", methods=["GET", "POST"])
def batsman():
    prediction   = None
    players      = batsmen.to_dict(orient='records')
    selected     = None

    if request.method == "POST":
        if request.form.get("player"):
            p    = batsmen[batsmen['Player'] == request.form["player"]].iloc[0]
            vals = [[p['Matches'], p['Runs'], p['Average'],
                     p['StrikeRate'], p['Centuries'], p['Experience']]]
            selected = p['Player']
        else:
            vals = [[float(request.form.get(x) or 0)
                     for x in ['matches', 'runs', 'average', 'sr', 'cent', 'exp']]]
            selected = "Custom Player"

        prediction = round(max(0.0, batsman_model.predict(vals)[0]), 2)

    return render_template("batsman.html", players=players,
                           prediction=prediction, selected_player=selected)


@app.route("/allrounder", methods=["GET", "POST"])
def allrounder():
    prediction   = None
    players      = allrounders.to_dict(orient='records')
    selected     = None

    if request.method == "POST":
        if request.form.get("player"):
            p    = allrounders[allrounders['Player'] == request.form["player"]].iloc[0]
            vals = [[p['Matches'], p['Runs'], p['Average_Bat'], p['StrikeRate_Bat'],
                     p['Fifties'], p['Wickets'], p['Economy'],
                     p['Average_Bowl'], p['StrikeRate_Bowl'], p['Experience']]]
            selected = p['Player']
        else:
            vals = [[float(request.form.get(x) or 0)
                     for x in ['matches', 'runs', 'avg_bat', 'sr_bat',
                                'fifties', 'wickets', 'economy', 'avg_bowl', 'sr_bowl', 'exp']]]
            selected = "Custom Player"

        prediction = round(max(0.0, allrounder_model.predict(vals)[0]), 2)

    return render_template("allrounder.html", players=players,
                           prediction=prediction, selected_player=selected)


if __name__ == "__main__":
    app.run(debug=True)
