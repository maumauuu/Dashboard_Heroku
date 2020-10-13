#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go



df = pd.read_csv("data/timesData.csv")
available_indicators = df.columns.unique()


def generate_table( max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(min(len(df), max_rows))
        ])
    ])



def choose_year():
    return html.Div([
        dcc.Location(id='page',refresh=False),
        dcc.Dropdown(
            id='year-dropdown',
            options=[
                {'label': '2011', 'value': 2011},
                {'label': '2012', 'value': 2012},
                {'label': '2013', 'value': 2013}
            ],
            value=2012,
        ),
        dcc.RadioItems(
            id='abs-choice',
            options=[{'label': i, 'value': i }for i in available_indicators],
            value='female_male_ratio',
            labelStyle={'display': 'inline-block'}
        ),
        html.Div(id='dd-output-container')

    ])



def update_layout(year,x_name='female_male_ratio'):
    filtered_df = df[df.year == year].iloc[:50, :];
    return html.Div([dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=filtered_df[filtered_df['country'] == i][x_name],
                        y=filtered_df[filtered_df['country'] == i]['num_students'],
                        text=filtered_df[filtered_df['country'] == i]['university_name'],
                        mode='markers',
                        opacity=0.8,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=i
                    ) for i in filtered_df.country.unique()
                ],
                'layout': go.Layout(
                    xaxis={'title': x_name},
                    yaxis={'title': 'Life Expectancy'},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0.0, 'y': 1},
                    hovermode='closest'
                )
            }
        )
    ])

