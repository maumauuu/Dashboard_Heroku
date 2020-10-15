#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dataReader import create_table, reg_np

df = pd.read_csv("data/timesData.csv")
available_indicators = df.columns.unique()
cd = pd.read_csv("data/carData.csv")

Qs = ['Q1', 'Q2', 'Q4', 'Q5', 'Q51', 'Q6']
Ts = ['Data Snapshot', 'Data Exploration', 'Data Visualization', 'Linear regression', 'My linear Regression', 'SVM']


def menu():
    return html.Div([
        html.Div([

            dcc.Location(id='page', refresh=False),
            html.H3('Menu'),
            dcc.RadioItems(
                id='exo-choice',
                options=[{'label': l, 'value': v} for l, v in zip(Ts, Qs)],
                value='Q1',
                labelStyle={'display': 'block'}
            )], style={'width': '20%', 'display': 'inline-block'}),
        html.Div(id='menu-output-container',
                 style={'width': '80%', 'display': 'inline-block'})
    ])


def bar_chart():
    cdd = cd[['Car_Name', 'Transmission', 'Selling_Price', 'Fuel_Type']]
    pv = cdd.pivot_table(index=['Car_Name'], columns=['Transmission'], values=['Selling_Price'], fill_value=0)
    pv2 = cdd.pivot_table(index=['Car_Name'], columns=['Fuel_Type'], values=['Selling_Price'], fill_value=0)

    print(pv)
    trace1 = go.Bar(x=pv.index, y=pv[('Selling_Price', 'Manual')], name='Manual')
    trace2 = go.Bar(x=pv.index, y=pv[('Selling_Price', 'Automatic')], name='Automatic')
    trace3 = go.Bar(x=pv2.index, y=pv2[('Selling_Price', 'Diesel')], name='Diesel')
    trace4 = go.Bar(x=pv2.index, y=pv2[('Selling_Price', 'Petrol')], name='Petrol')
    trace5 = go.Bar(x=pv2.index, y=pv2[('Selling_Price', 'CNG')], name='CNG')

    return html.Div(children=[
        html.H1(children='Transmission type'),

        dcc.Graph(
            id='example-graph',
            figure={
                'data': [trace1, trace2, trace3, trace4, trace5],
                'layout':
                    go.Layout(title='Selling Price according to the transmission and Fuel Type', barmode='stack')
            }),
        html.P(className='box', children='''\n\n In this bar chart, we can see the price of each vehicle according to 
        the transmission type and Fuel type. We can see that on the lower range of Cars there is not much choice for 
        the same car e.g Activia 4g have only one type of fuel and one of transmission. 
        
        On the contrary,
        ''')
    ])


def generate_table():
    create_table()
    return html.Div([
        dcc.Dropdown(
            id='show_data',
            options=[
                {'label': 'Numpy', 'value': 1},
                {'label': 'Scipy', 'value': 2},
                {'label': 'Sickit Learn', 'value': 3},
                {'label': 'Sickit Learn multiple', 'value': 4}
            ],
            value=0
        ),
        dcc.Graph(
            id='data_dropdown')
    ])


def plot_regr(val):
    if val == 1:
        return reg_np()
    if val == 2:
        return
    if val == 3:
        return
    if val == 4:
        return


def table(df_, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df_])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df_.iloc[i][col]) for col in df_
            ]) for i in range(min(len(df_), max_rows))
        ])
    ])


def analyse_data(val):
    if val == 'Q1':
        return table(cd)
    if val == 'Q2':
        # print(cd.describe())
        cd_d = cd.describe()
        cd_d['Stats'] = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        header_list = ['Stats'] + [c for c in cd.describe()]
        # print(header_list)
        cd_d = cd_d.reindex(columns=header_list)
        # print('!!!!!!!!!!2\n',cd_d)
        return table(cd_d)
    elif val == 'Q4':
        return bar_chart()
    elif val == 'Q5':
        return generate_table()
    elif val == 'Q51':
        return
    elif val == 'Q6':
        return
    else:
        return table(cd)


def choose_year():
    return html.Div([
        dcc.Location(id='page', refresh=False),
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
            options=[{'label': i, 'value': i} for i in available_indicators],
            value='female_male_ratio',
            labelStyle={'display': 'inline-block'}
        ),
        html.Div(id='dd-output-container')

    ])


def draw_seaborn():
    return html.Div()


def update_layout(year, x_name='female_male_ratio'):
    filtered_df = df[df.year == year].iloc[:50, :]
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
