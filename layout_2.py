#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dataReader import data_vis, create_table, requete_price,reg_np, reg_sp,reg_sk,reg_sk_multiple,my_reg, svm_
import seaborn as sns

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
        html.P(className='box', children=''' \n\n In this bar chart, we can see the price of each vehicle according to the transmission type and Fuel type.
        We can see that on the lower range of Cars there is not much choice for the same car e.g Activia 4g have only one type of fuel and one 
        of transmission.

        On the contrary,
        ''')
    ])


def generate_table(max_rows=10):
    return html.Div([
        dcc.Dropdown(
            id='show_data',
            options=[
                {'label': 'Numpy', 'value': 1},
                {'label': 'Scipy', 'value': 2},
                {'label': 'Sickit Learn', 'value': 3},
                {'label': 'Sickit Learn multiple', 'value': 4}
            ],
            value=1
        ),

        html.Div([
            dcc.Graph(id='data_dropdown'),
            html.P(id='comment')
            ])
    ])


def plot_regr(val):
    if val == 1:
        return reg_np()
    if val == 2:
        return reg_sp()
    if val == 3:
        return reg_sk()
    if val == 4:
        return reg_sk_multiple()
    else:
        return

def gen_plot_svm():
    fig, com =svm_()
    return html.Div([
        dcc.Graph(id='plot-svm',figure=fig),
        html.P(id='comment', children=com)
    ])

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
        cd_d = cd.describe()
        cd_d['Stats'] = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        header_list = ['Stats'] + [c for c in cd.describe()]
        cd_d = cd_d.reindex(columns=header_list)
        return table(cd_d)
    elif val == 'Q4':
        return bar_chart()
    elif val == 'Q5':
        return generate_table()
    elif val == 'Q51':
        return
    elif val == 'Q6':
        print('\n\nq6')
        return gen_plot_svm()
    else:
        return table(cd)

