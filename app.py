import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Load both datasets
df = pd.read_csv("event_df_enriched.csv", parse_dates=["Earnings_Date", "Date"])
top_after_hours_df = pd.read_csv("top_after_hours_df.csv", parse_dates=["Date"])

# Clean and prep after-hours dataset
top_after_hours_df = top_after_hours_df.sort_values(by='Date')
top_after_hours_df = top_after_hours_df.drop_duplicates(subset=['Ticker', 'Date'], keep='first')
top_after_hours_df = top_after_hours_df.dropna(subset=['After_Hours_Market_Change'])


model_df = pd.read_csv("combined_model_performance.csv")

global_results_df = pd.read_csv("global_model_predictions.csv")
global_metrics_df = pd.read_csv("global_model_metrics.csv")

# Load cumulative model performance data
model_comparison_df = pd.read_csv("cumulative_result_df.csv")


print("📌 Available Models:", global_metrics_df["Model"].unique())


# App setup

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "After Hours Stock Analysis Dashboard"

# Layout
app.layout = dbc.Container([
    html.H1("After Hours Stock Analysis Dashboard", className="text-center my-4"),


    dcc.Tabs([
        dcc.Tab(label="Comparative Analysis", children=[
            dcc.Tabs([
            dcc.Tab(label="After-Hours Movers", children=[
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        html.Label("Date Range"),
                        html.Br(),
                        dcc.DatePickerRange(
                            id='date-range',
                            min_date_allowed=top_after_hours_df['Date'].min(),
                            max_date_allowed=top_after_hours_df['Date'].max(),
                            start_date=top_after_hours_df['Date'].min(),
                            end_date=top_after_hours_df['Date'].max()
                        )
                    ], width="auto", className="me-4"),

                    dbc.Col([
                        html.Label("Sector"),
                        html.Br(),
                        dcc.Dropdown(
                            id='sector-dropdown',
                            options=[{'label': s, 'value': s} for s in sorted(top_after_hours_df['Sector'].dropna().unique())],
                            value=None,
                            placeholder="All Sectors",
                            clearable=True,
                            style={'width': '300px'}
                        )
                    ], width="auto", className="me-4"),

                    dbc.Col([
                        html.Label("Top N"),
                        html.Br(),
                        dcc.Input(
                            id='top-n',
                            type='number',
                            min=1,
                            max=len(top_after_hours_df['Ticker'].unique()),
                            value=10,
                            debounce=True,
                            style={'width': '100px'}
                        )
                    ], width="auto", className="me-4")
                ], className="mb-4"),
                dcc.Graph(id='bar-top-movers'),
                html.Div([
                    html.H4("🔎 Details"),
                    html.Div(id='bar-hover-info', style={
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'backgroundColor': '#f9f9f9',
                        'marginTop': '10px'
                    })
                ]),
                html.H3("Inspect Ticker"),
                dcc.Dropdown(id='ticker-dropdown', placeholder="Select a ticker..."),
                dcc.Graph(id='line-chart')
            ])

        ])
        ]),
        dcc.Tab(label="Event Analysis", children=[
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
        ]),
        dcc.Tab(label="Model Performance", children=[
            html.Br(),
            html.Div([
                html.H3("Model Evaluation over 40+ Top Tickers"),
                dcc.RadioItems(
                    id="global-model-selector",
                    options=[
                        {"label": "Random Forest", "value": "Random Forest"},
                        {"label": "XGBoost", "value": "XGBoost"},
                        {"label": "LSTM", "value": "LSTM"}
                    ],
                    value="XGBoost",
                    labelStyle={"display": "inline-block", "marginRight": "20px"}
                ),
                dcc.Graph(id="global-model-plot"),
                html.Br(),
                html.H3("Window Based Comparison:"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Filter by Model"),
                        dcc.Dropdown(
                            id="filter-model",
                            options=[{"label": m, "value": m} for m in model_comparison_df["Model"].unique()],
                            placeholder="Select a model",
                            clearable=True
                        )
                    ], width=3),

                    dbc.Col([
                        html.Label("Filter by Window Size"),
                        dcc.Dropdown(
                            id="filter-window",
                            options=[{"label": int(w), "value": int(w)} for w in model_comparison_df["Window"].unique()],
                            placeholder="Select window size",
                            clearable=True
                        )
                    ], width=3),
                ], className="mb-3"),
                html.Div(id="model-comparison-table"),
                html.Label("Individual Ticker Evaluation Metric:"),
                dcc.RadioItems(
                    id="metric-selector",
                    options=[
                        {"label": "RMSE", "value": "RMSE"},
                        {"label": "R² Score", "value": "R2"}
                    ],
                    value="RMSE",
                    labelStyle={"display": "inline-block", "marginRight": "20px"}
                ),
                html.Hr()
            ]),
            dcc.Graph(id="model-performance-chart")
        ])

    ])
], fluid=True)


# === Callbacks for tabs ===

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
    fig.add_trace(go.Scatter(x=grouped["Days_From_Earnings"], y=grouped["Regular_Change%"], mode="lines+markers", name="Regular Hours"))
    fig.add_trace(go.Scatter(x=grouped["Days_From_Earnings"], y=grouped["After_Hours_Change%"], mode="lines+markers", name="After Hours"))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Earnings Day")
    fig.update_layout(title="Price Movement Around Earnings", xaxis_title="Days From Earnings", yaxis_title="Change (%)")
    return fig
@app.callback(
    Output("eps-impact-regular", "figure"),
    Output("eps-impact-afterhours", "figure"),
    Input("surprise-filter", "value")
)
def update_eps_impact(surprise_filter):
    dff = df[df["Days_From_Earnings"].between(-3, 3)]
    dff = dff[dff["EPS_Surprise_Clean"] > 0] if surprise_filter == "positive" else dff[dff["EPS_Surprise_Clean"] <= 0]
    grouped = dff.groupby("Days_From_Earnings").agg({
        "Regular_Change%": "mean",
        "After_Hours_Change%": "mean"
    }).reset_index()
    fig1 = px.line(grouped, x="Days_From_Earnings", y="Regular_Change%", title="Regular Hours Change (%)")
    fig2 = px.line(grouped, x="Days_From_Earnings", y="After_Hours_Change%", title="After Hours Change (%)")
    for fig in [fig1, fig2]: fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig1, fig2

@app.callback(
    Output("sector-regular", "figure"),
    Output("sector-afterhours", "figure"),
    Input("sector-filter", "value")
)
def update_sector_charts(sectors):
    dff = df[df["Days_From_Earnings"].between(-3, 3)]
    if sectors: dff = dff[dff["Sector"].isin(sectors)]
    grouped = dff.groupby(["Days_From_Earnings", "Sector"]).agg({
        "Regular_Change%": "mean",
        "After_Hours_Change%": "mean"
    }).reset_index()
    fig1 = px.line(grouped, x="Days_From_Earnings", y="Regular_Change%", color="Sector", title="Regular Hours Change")
    fig2 = px.line(grouped, x="Days_From_Earnings", y="After_Hours_Change%", color="Sector", title="After Hours Change")
    for fig in [fig1, fig2]: fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig1, fig2

@app.callback(
    Output("heatmap-chart", "figure"),
    Input("heatmap-chart", "id")  # dummy trigger
)
def update_heatmap(_):
    # --- Bin EPS Surprise into meaningful categories ---
    bins = [-float("inf"), 0, 5, 10, 20, float("inf")]
    labels = ["<=0%", "0–5%", "5–10%", "10–20%", ">20%"]
    df["EPS_Surprise_Bin"] = pd.cut(df["EPS_Surprise_%"], bins=bins, labels=labels)

    # --- Group by Sector and Surprise Bin ---
    grouped = df.groupby(["Sector", "EPS_Surprise_Bin"]).agg({
        "Regular_Change%": "mean",
        "After_Hours_Change%": "mean"
    }).reset_index()

    # Ensure consistent bin order
    grouped["EPS_Surprise_Bin"] = pd.Categorical(grouped["EPS_Surprise_Bin"], categories=labels, ordered=True)

    # --- Pivot for heatmaps ---
    pivot_regular = grouped.pivot(index="Sector", columns="EPS_Surprise_Bin", values="Regular_Change%")
    pivot_afterhours = grouped.pivot(index="Sector", columns="EPS_Surprise_Bin", values="After_Hours_Change%")

    # --- Plot side-by-side heatmaps ---
    import plotly.subplots as sp
    import plotly.express as px

    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=("Regular Hours Return (%)", "After-Hours Return (%)"),
        shared_yaxes=True
    )

    # Create heatmaps with fixed color scale range
    heatmap1 = px.imshow(
        pivot_regular,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        zmin=-5, zmax=5
    )
    heatmap2 = px.imshow(
        pivot_afterhours,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        zmin=-5, zmax=5
    )

    # Add both to subplot
    for trace in heatmap1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in heatmap2.data:
        fig.add_trace(trace, row=1, col=2)

    # Final layout
    fig.update_layout(
        title="Sector-wise Avg Return by EPS Surprise Bin",
        height=600,
        width=1100,
        coloraxis_colorbar=dict(title="Return (%)"),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


@app.callback(
    Output('bar-top-movers', 'figure'),
    Output('ticker-dropdown', 'options'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('top-n', 'value'),
    Input('sector-dropdown', 'value')
)
def update_bar_chart(start_date, end_date, top_n, selected_sector):
    filtered = top_after_hours_df.query("Date >= @start_date and Date <= @end_date")
    if selected_sector:
        filtered = filtered[filtered["Sector"] == selected_sector]
    filtered = filtered[filtered["After_Hours_Market_Change"] < 100]
    top_movers = filtered.sort_values(by='After_Hours_Market_Change', ascending=False).head(top_n)
    fig = px.bar(top_movers, x='Ticker', y='After_Hours_Market_Change', color='Sector', title="Top After-Hours Movers")
    options = [{'label': t, 'value': t} for t in top_movers['Ticker'].unique()]
    return fig, options

@app.callback(
    Output('bar-hover-info', 'children'),
    Input('bar-top-movers', 'hoverData')
)
def display_hover_info(hoverData):
    if not hoverData: return "Hover over a bar to see details."
    ticker = hoverData['points'][0]['x']
    row = top_after_hours_df[top_after_hours_df['Ticker'] == ticker].sort_values('Date', ascending=False).head(1).iloc[0]
    return html.Div([
        html.P(f"Ticker: {row['Ticker']}"), html.P(f"Sector: {row['Sector']}"),
        html.P(f"Date: {row['Date'].date()}"), html.P(f"Open: {row['Open']:.2f}"),
        html.P(f"High: {row['High']:.2f}"), html.P(f"Low: {row['Low']:.2f}"),
        html.P(f"Close: {row['Close']:.2f}"), html.P(f"Volume: {int(row['Volume']):,}"),
        html.P(f"After-Hours % Change: {row['After_Hours_Market_Change']:.2f}%")
    ])

@app.callback(
    Output('line-chart', 'figure'),
    Input('ticker-dropdown', 'value')
)
def update_line_chart(ticker):
    if not ticker:
        return px.line(title="Select a ticker to see Close Price Trend")
    df_ticker = top_after_hours_df[top_after_hours_df['Ticker'] == ticker]
    return px.line(df_ticker, x='Date', y='Close', title=f"{ticker} Close Price Over Time")

@app.callback(
    Output("model-performance-chart", "figure"),
    Input("metric-selector", "value")
)
def update_model_chart(selected_metric):
    if selected_metric not in ["RMSE", "R2"]:
        selected_metric = "RMSE"
    
    fig = px.bar(
        model_df,
        x="Ticker",
        y=selected_metric,
        color="Model",
        barmode="group",
        title=f"📊 {selected_metric} Comparison Across Models",
        height=500
    )
    return fig

@app.callback(
    Output("global-model-plot", "figure"),
    Input("global-model-selector", "value")
)
def update_global_model_tab(selected_model):

    print("📌 Selected:", selected_model)

    model_df = global_results_df[global_results_df["Model"] == selected_model]
    #metrics_row = global_metrics_df[global_metrics_df["Model"] == selected_model.capitalize()].iloc[0]
    # Filter the DataFrame
    filtered = global_metrics_df[global_metrics_df["Model"] == selected_model]

    # Check if we have any results
    if filtered.empty:
        return go.Figure(), html.Div("⚠️ No metrics found for this model.")

    # Safe access
    metrics_row = filtered.iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=model_df["True"], mode="lines", name="Actual", line=dict(color="black")
    ))
    fig.add_trace(go.Scatter(
        y=model_df["Predicted"], mode="lines", name="Predicted",
        line=dict(color="green" if selected_model == "rf" else "blue")
    ))

    fig.update_layout(
        title=f"{selected_model.upper()} | 📉 RMSE: {metrics_row['RMSE']:.4f} | 📈 R² Score: {metrics_row['R2']:.4f}",
        title_x=0.5,
        margin=dict(t=80)
    )

    return fig

from dash import dash_table

@app.callback(
    Output("model-comparison-table", "children"),
    Input("filter-model", "value"),
    Input("filter-window", "value")
)
def update_model_comparison_table(selected_model, selected_window):
    filtered_df = model_comparison_df.copy()

    if selected_model:
        filtered_df = filtered_df[filtered_df["Model"] == selected_model]
    if selected_window:
        filtered_df = filtered_df[filtered_df["Window"] == selected_window]

    return dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in filtered_df.columns],
        data=filtered_df.to_dict("records"),
        style_cell={
            "textAlign": "center",
            "padding": "5px",
        },
        style_header={
            "backgroundColor": "lightgrey",
            "fontWeight": "bold"
        },
        style_table={"overflowX": "auto"},
        page_size=10,
        sort_action="native"
    )




if __name__ == "__main__":
    app.run(debug=True)
