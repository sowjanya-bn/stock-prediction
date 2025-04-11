
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Load data
df = pd.read_csv("event_df_cleaned.csv", parse_dates=["Earnings_Date", "Date"])

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Earnings Event Dashboard"

# Layout with Tabs
app.layout = dbc.Container([
    html.H1("Earnings Event Interactive Dashboard", className="text-center my-4"),
    dcc.Tabs([
        dcc.Tab(label="Price Movement (±3 Days)", children=[
            html.Br(),
            dcc.Dropdown(
                id="ticker-filter",
                options=[{"label": t, "value": t} for t in sorted(df["Ticker"].unique())],
                multi=True,
                placeholder="Select one or more tickers"
            ),
            dcc.Graph(id="price-movement-chart")
        ]),
        dcc.Tab(label="EPS Surprise Impact", children=[
            html.Br(),
            dcc.RadioItems(
                id="surprise-filter",
                options=[
                    {"label": "Positive Surprise", "value": "positive"},
                    {"label": "Negative/No Surprise", "value": "negative"}
                ],
                value="positive",
                labelStyle={"display": "inline-block", "marginRight": "20px"}
            ),
            dcc.Graph(id="eps-impact-regular"),
            dcc.Graph(id="eps-impact-afterhours")
        ]),
        dcc.Tab(label="Sector Breakdown", children=[
            html.Br(),
            dcc.Dropdown(
                id="sector-filter",
                options=[{"label": s, "value": s} for s in sorted(df["Sector"].unique())],
                multi=True,
                placeholder="Select sectors to display"
            ),
            dcc.Graph(id="sector-regular"),
            dcc.Graph(id="sector-afterhours")
        ]),
        dcc.Tab(label="EPS Surprise Heatmap", children=[
            html.Br(),
            dcc.Graph(id="heatmap-chart")
        ])
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output("price-movement-chart", "figure"),
    Input("ticker-filter", "value")
)
def update_price_movement(tickers):
    dff = df[df["Days_From_Earnings"].between(-3, 3)]
    if tickers:
        dff = dff[dff["Ticker"].isin(tickers)]
    grouped = dff.groupby("Days_From_Earnings").agg({
        "Regular_Change%": "mean",
        "After_Hours_Change%": "mean"
    }).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["Days_From_Earnings"], y=grouped["Regular_Change%"],
                             mode="lines+markers", name="Regular Hours Change (%)"))
    fig.add_trace(go.Scatter(x=grouped["Days_From_Earnings"], y=grouped["After_Hours_Change%"],
                             mode="lines+markers", name="After Hours Change (%)"))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Earnings Day")
    fig.update_layout(title="Average Price Movement Around Earnings (±3 Days)",
                      xaxis_title="Days From Earnings", yaxis_title="Average % Change")
    return fig

@app.callback(
    Output("eps-impact-regular", "figure"),
    Output("eps-impact-afterhours", "figure"),
    Input("surprise-filter", "value")
)
def update_eps_impact(surprise_filter):
    dff = df[df["Days_From_Earnings"].between(-3, 3)]
    if surprise_filter == "positive":
        dff = dff[dff["EPS_Surprise"] > 0]
    else:
        dff = dff[dff["EPS_Surprise"] <= 0]
    grouped = dff.groupby("Days_From_Earnings").agg({
        "Regular_Change%": "mean",
        "After_Hours_Change%": "mean"
    }).reset_index()
    fig1 = px.line(grouped, x="Days_From_Earnings", y="Regular_Change%", title="Regular Hours Change (%)")
    fig2 = px.line(grouped, x="Days_From_Earnings", y="After_Hours_Change%", title="After Hours Change (%)")
    for fig in [fig1, fig2]:
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig1, fig2

@app.callback(
    Output("sector-regular", "figure"),
    Output("sector-afterhours", "figure"),
    Input("sector-filter", "value")
)
def update_sector_charts(sectors):
    dff = df[df["Days_From_Earnings"].between(-3, 3)]
    if sectors:
        dff = dff[dff["Sector"].isin(sectors)]
    grouped = dff.groupby(["Days_From_Earnings", "Sector"]).agg({
        "Regular_Change%": "mean",
        "After_Hours_Change%": "mean"
    }).reset_index()
    fig1 = px.line(grouped, x="Days_From_Earnings", y="Regular_Change%", color="Sector", title="Regular Hours Change by Sector")
    fig2 = px.line(grouped, x="Days_From_Earnings", y="After_Hours_Change%", color="Sector", title="After Hours Change by Sector")
    for fig in [fig1, fig2]:
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig1, fig2

@app.callback(
    Output("heatmap-chart", "figure"),
    Input("heatmap-chart", "id")  # Dummy input to trigger on page load
)
def update_heatmap(_):
    heatmap_data = df[df["EPS_Surprise_Bin"] != "<=0%"]
    grouped = heatmap_data.groupby(["Sector", "EPS_Surprise_Bin"]).agg({
        "Regular_Change%": "mean"
    }).reset_index()
    pivoted = grouped.pivot(index="Sector", columns="EPS_Surprise_Bin", values="Regular_Change%")
    fig = px.imshow(pivoted, text_auto=True, aspect="auto", color_continuous_scale="Greens",
                    title="Avg Return (%) by Sector and EPS Surprise Threshold")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
