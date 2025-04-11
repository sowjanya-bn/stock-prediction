import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Load both datasets
df = pd.read_csv("event_df_cleaned.csv", parse_dates=["Earnings_Date", "Date"])
top_after_hours_df = pd.read_csv("top_after_hours_df.csv", parse_dates=["Date"])

# Clean and prep after-hours dataset
top_after_hours_df = top_after_hours_df.sort_values(by='Date')
top_after_hours_df = top_after_hours_df.drop_duplicates(subset=['Ticker', 'Date'], keep='first')
top_after_hours_df = top_after_hours_df.dropna(subset=['After_Hours_Market_Change'])

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
    dff = dff[dff["EPS_Surprise"] > 0] if surprise_filter == "positive" else dff[dff["EPS_Surprise"] <= 0]
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
    # Filter and group
    heatmap_data = df[df["EPS_Surprise_Bin"] != "<=0%"]
    grouped = heatmap_data.groupby(["Sector", "EPS_Surprise_Bin"]).agg({
        "Regular_Change%": "mean"
    }).reset_index()

    # Ensure bin order
    bin_order = ['>0%', '>5%', '>10%']
    grouped["EPS_Surprise_Bin"] = pd.Categorical(grouped["EPS_Surprise_Bin"], categories=bin_order, ordered=True)

    # Pivot for heatmap
    pivoted = grouped.pivot(index="Sector", columns="EPS_Surprise_Bin", values="Regular_Change%")

    # Plot
    fig = px.imshow(
        pivoted,
        text_auto=True,
        color_continuous_scale="Greens",
        aspect="auto",
        title="Sector-wise Avg Return by EPS Surprise Bin"
    )
    fig.update_layout(xaxis_title="EPS Surprise Bin", yaxis_title="Sector")
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

if __name__ == "__main__":
    app.run(debug=True)
