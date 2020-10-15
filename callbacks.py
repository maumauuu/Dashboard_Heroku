#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dash.dependencies import Input, Output
from layout_2 import update_layout, analyse_data, plot_regr
from app import app


@app.callback(
    Output('dd-output-container', 'children'),
    [Input('year-dropdown', 'value'),
     Input('abs-choice', 'value')])
def display_value(year, indicator):
    """changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'year-dropdown'in changed_id:
        return update_layout(year)
    elif 'abs-choice'in changed_id:"""
    return update_layout(year, indicator)


@app.callback(Output('data_dropdown', 'figure'),
              [Input('show_data', 'value')])
def show_data(value):
    return plot_regr(value)


@app.callback(Output('menu-output-container', 'children'),
              [Input('exo-choice', 'value')])
def show_data(value):
    return analyse_data(value)


@app.callback(Output('page', 'pathname'),
              [Input('year-dropdown', 'value')])
def change_url(year):
    return '/apps/' + str(year)
