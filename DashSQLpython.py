# Install Dash library if you haven't
# pip install dash

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

# Assume df is your DataFrame
# df = pd.read_csv("your_dataset.csv")

fig = px.bar(df_product, x='productLine', y='quantityInStock', title="Stock by Product Line")

app = dash.Dash()

app.layout = html.Div([
    dcc.Graph(
        id='stock-by-productLine',
        figure=fig
    )
])

# Import additional components
from dash.dependencies import Input, Output
import dash_table

# Create more figures
fig2 = px.box(df_product, x='productLine', y='MSRP', title="MSRP by Product Line")

# Initialize the Dash app
app = dash.Dash()

# Define the layout
app.layout = html.Div([
  # Dropdown for selecting metrics
  html.Label('Select Metric for Bar Plot:'),
  dcc.Dropdown(
    id='dropdown-metric',
    options=[
      {'label': 'Quantity In Stock', 'value': 'quantityInStock'},
      {'label': 'MSRP', 'value': 'MSRP'},
      {'label': 'Buy Price', 'value': 'buyPrice'}
    ],
    value='quantityInStock'
  ),

  # Graphs
  dcc.Graph(
    id='bar-plot'
  ),
  dcc.Graph(
    id='box-plot',
    figure=fig2
  ),

  # Data Table
  dash_table.DataTable(
    id='data-table',
    columns=[{"name": i, "id": i} for i in df_product.columns],
    data=df_product.to_dict('records'),
  )
])


# Define callback to update bar plot
@app.callback(
  Output('bar-plot', 'figure'),
  [Input('dropdown-metric', 'value')]
)
def update_bar_chart(selected_metric):
  return px.bar(df_product, x='productLine', y=selected_metric, title=f"{selected_metric} by Product Line")


# Run the app
if __name__ == '__main__':
  app.run_server(debug=True)
