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

if __name__ == '__main__':
    app.run_server(debug=True)
