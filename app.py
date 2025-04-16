import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import statsmodels.api as sm
import plotly.express as px

# Load the recoded CSV file (make sure it includes numeric values for the recoded variables)
df = pd.read_csv('soil_pollution_diseases_recoded.csv')

# Define the dependent variable and the list of independent variables.
dependent_var = "Disease_Severity"
independent_vars = [
    "Pollutant_Concentration_mg_kg",
    "Soil_pH",
    "Soil_Organic_Matter_%",
    "Temperature_C",
    "Rainfall_mm",
    "Humidity_%",
    "Pollutant_Type",
    "Nearby_Industry",
    "Farming_Practice",
    "Age_Group_Affected"
]

# Initialize the Dash app.
app = dash.Dash(__name__)
app.title = "Soil Pollution & Health Impacts Regression Dashboard"

# Define the layout.
app.layout = html.Div([
    html.H1("Soil Pollution & Health Impacts Regression Dashboard", style={'textAlign': 'center'}),
    html.P(
        "This dashboard regresses Disease_Severity (dependent variable) on the following independent variables: "
        "Pollutant_Concentration_mg_kg, Soil_pH, Soil_Organic_Matter_%, Temperature_C, Rainfall_mm, Humidity_%, "
        "Pollutant_Type, Nearby_Industry, Farming_Practice, and Age_Group_Affected."
    ),

    # Dropdown to choose an independent variable for the box plot.
    html.Div([
        html.Label("Select independent variable for box plot:"),
        dcc.Dropdown(
            id="boxplot-var-dropdown",
            options=[{"label": var, "value": var} for var in independent_vars],
            value="Pollutant_Concentration_mg_kg",
            clearable=False
        )
    ], style={'width': '50%', 'padding': '10px'}),

    dcc.Graph(id="box-plot"),

    html.H2("Full Model Regression Summary"),
    html.Pre(id="regression-summary", style={'overflowX': 'auto'}),

    html.H2("Full Model Coefficients"),
    dash_table.DataTable(
        id='regression-coef-table',
        columns=[
            {"name": "Variable", "id": "Variable"},
            {"name": "Coefficient", "id": "Coefficient"},
            {"name": "P-value", "id": "P-value"}
        ],
        style_table={'width': '60%', 'margin': 'auto'},
        style_cell={'textAlign': 'center'}
    ),

    html.H2("Omitted Variable Bias Analysis for 'Pollutant_Concentration_mg_kg'"),
    html.P(
        "For each independent variable (other than Pollutant_Concentration_mg_kg), the table "
        "shows how omitting that variable changes the coefficient of Pollutant_Concentration_mg_kg."
    ),
    dash_table.DataTable(
        id='ovb-table',
        columns=[
            {"name": "Omitted Variable", "id": "Omitted_Variable"},
            {"name": "Full Model Coefficient", "id": "Full_Coefficient"},
            {"name": "Reduced Model Coefficient", "id": "Reduced_Coefficient"},
            {"name": "Absolute Change", "id": "Absolute_Change"},
            {"name": "Percentage Change (%)", "id": "Percentage_Change"}
        ],
        style_table={'width': '80%', 'margin': 'auto'},
        style_cell={'textAlign': 'center'}
    )
])


# Callback to update the dashboard components when the user selects a new independent variable for the box plot.
@app.callback(
    [Output("box-plot", "figure"),
     Output("regression-summary", "children"),
     Output("regression-coef-table", "data"),
     Output("ovb-table", "data")],
    [Input("boxplot-var-dropdown", "value")]
)
def update_dashboard(boxplot_var):
    # Prepare the dataset: select the dependent and independent variables, then drop rows with missing values.
    df_subset = df[[dependent_var] + independent_vars].dropna().copy()

    # Convert columns to numeric (non-numeric strings become NaN, then drop the affected rows).
    for col in [dependent_var] + independent_vars:
        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
    df_subset = df_subset.dropna()

    if df_subset.empty:
        empty_fig = {"data": [], "layout": {"title": "No Data Available"}}
        return empty_fig, "Insufficient numeric data available after cleaning.", [], []

    # Create a box plot:
    # Here we group the data by Disease_Severity (x-axis) and show the distribution of the chosen independent variable (y-axis).
    fig = px.box(
        df_subset,
        x=dependent_var,
        y=boxplot_var,
        points="all",  # Show all points overlaid on the box plot
        title=f"Distribution of {boxplot_var} by Disease Severity"
    )

    # Build the full regression model using all independent variables.
    X_full = df_subset[independent_vars]
    y = df_subset[dependent_var]
    X_full = sm.add_constant(X_full)
    model_full = sm.OLS(y, X_full).fit()
    summary_text = model_full.summary().as_text()
    coef_df = pd.DataFrame({
        "Variable": model_full.params.index,
        "Coefficient": model_full.params.values,
        "P-value": model_full.pvalues.values
    })
    coef_data = coef_df.to_dict("records")

    # Omitted Variable Bias (OVB) Analysis for Pollutant_Concentration_mg_kg:
    main_var = "Pollutant_Concentration_mg_kg"
    full_coef_main = model_full.params[main_var]
    ovb_results = []

    for var in independent_vars:
        if var == main_var:
            continue
        reduced_vars = [v for v in independent_vars if v != var]
        X_reduced = df_subset[reduced_vars]
        X_reduced = sm.add_constant(X_reduced)
        model_reduced = sm.OLS(y, X_reduced).fit()
        reduced_coef_main = model_reduced.params[main_var]
        abs_change = abs(reduced_coef_main - full_coef_main)
        perc_change = (abs_change / abs(full_coef_main) * 100) if full_coef_main != 0 else None
        ovb_results.append({
            "Omitted_Variable": var,
            "Full_Coefficient": full_coef_main,
            "Reduced_Coefficient": reduced_coef_main,
            "Absolute_Change": abs_change,
            "Percentage_Change": perc_change
        })

    return fig, summary_text, coef_data, ovb_results


if __name__ == '__main__':
    app.run_server(debug=True)
