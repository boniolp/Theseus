import numpy as np

import base64

import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import  pandas as pd

# Plotly import
import plotly.graph_objs as go
import os
import ast
#from background import *

########################################################################
################################# DATA #################################
########################################################################
methods_key = ['IFOREST','LOF','MP','NORMA','IFOREST1','HBOS','OCSVM','PCA','AE','CNN','LSTM','POLY']
methods_key_measures = ['IFOREST','LOF','MP','NORMA','IFOREST1','HBOS','OCSVM','PCA','POLY']
measures_key = ['Precision@k','Precision','Recall','F','Rprecision','Rrecall','RF','AUC_ROC','AUC_PR']#,'R_AUC_ROC','R_AUC_PR','VUS_ROC','VUS_PR'

path_top_dataseries = "data/benchmark_new/"
path_top_anoamly_score = "data/scores_new/"

all_file_folder = {}
for folder in os.listdir(path_top_dataseries):
    if "." not in folder:
        all_files = [] 
        for file in os.listdir(path_top_dataseries+folder):
            all_files.append('{}_robustness.txt'.format(file.replace('.zip','')))
        all_file_folder[folder] = all_files

df = pd.read_csv('data/mergedTable_AUC_PR.csv')


dirname_lag =  'data/robustness_results/result_data_aggregated_lag/'
dirname_noise =  'data/robustness_results/result_data_aggregated_noise/'
dirname_percentage =  'data/robustness_results/result_data_aggregated_percentage/'


def generate_dict(list_file):
    all_dict_lag = []
    for file_s in os.listdir(dirname_lag):
        if file_s in list_file:
            f  = ast.literal_eval(open(dirname_lag+file_s).read()[:-1])
            all_dict_lag.append(f)

    all_dict_noise = []
    for file_s in os.listdir(dirname_noise):
        if file_s in list_file:
            f  = ast.literal_eval(open(dirname_noise+file_s).read()[:-1])
            all_dict_noise.append(f)

    all_dict_percentage = []
    for file_s in os.listdir(dirname_percentage):
        if file_s in list_file:
            f  = ast.literal_eval(open(dirname_percentage+file_s).read()[:-1])
            all_dict_percentage.append(f)
    return all_dict_lag,all_dict_noise,all_dict_percentage


def group_dict(all_dict_lag):
    d_lag = {}
    for k in list(all_dict_lag[0].keys())[::-1]:
        d_lag[k] = tuple(d[k] for d in all_dict_lag)
    return d_lag

all_dict = {}
for list_file_key in all_file_folder.keys():
    all_dict_lag,all_dict_noise,all_dict_percentage = generate_dict(all_file_folder[list_file_key])
    if len(all_dict_lag) > 1:
        all_dict[list_file_key] = {}
        all_dict[list_file_key]['lag'] = group_dict(all_dict_lag)
        all_dict[list_file_key]['noise'] = group_dict(all_dict_noise)
        all_dict[list_file_key]['percentage'] = group_dict(all_dict_percentage)

#global_dataframe = pd.DataFrame(columns=['folder','file_name','measure','type','value'])
rows_list = []
for folder in all_dict.keys():
    for measure in measures_key:
        if measure not in ['to_remove']:
            for i,(lag,noise,ratio) in enumerate(zip(all_dict[folder]['lag'][measure],all_dict[folder]['noise'][measure],all_dict[folder]['percentage'][measure])):
                    to_append_lag = {'folder': folder,
                                 'file_name': "{}_{}".format(folder,i),
                                 'measure': measure,
                                 'type': 'lag',
                                 'value': lag}
                    to_append_noise = {'folder': folder,
                                 'file_name': "{}_{}".format(folder,i),
                                 'measure': measure,
                                 'type': 'noise',
                                 'value': noise}
                    to_append_ratio = {'folder': folder,
                                 'file_name': "{}_{}".format(folder,i),
                                 'measure': measure,
                                 'type': 'ratio',
                                 'value': ratio}
                    rows_list.append(to_append_lag)
                    rows_list.append(to_append_noise)
                    rows_list.append(to_append_ratio)
global_dataframe = pd.DataFrame(rows_list)

########################################################################
############################ CONSTANTS #################################
########################################################################


image_filename_intro = './assets/Chapter3_illustration.png'
encoded_image_intro = base64.b64encode(open(image_filename_intro, 'rb').read())


image_filename_pd = './assets/parisDescartes-logo.png'
encoded_image_pd = base64.b64encode(open(image_filename_pd, 'rb').read())
image_filename_dn = './assets/dinoLogo.png'
encoded_image_dn = base64.b64encode(open(image_filename_dn, 'rb').read())
image_filename_lp = './assets/LIPADE.png' 
encoded_image_lp = base64.b64encode(open(image_filename_lp, 'rb').read())
image_filename_edf = './assets/edf-s2sound.png' 
encoded_image_edf = base64.b64encode(open(image_filename_edf, 'rb').read())

description = "Detection anomalies in time series have gained ample academic and industrial attention. \
	However, no comprehensive benchmark exists to evaluate time-series anomaly detection methods. \
	It is common to use (i) proprietary or synthetic data, often biased to support particular claims, \
	or (ii) a limited collection of publicly available datasets. Consequently, we often observe \
	methods performing exceptionally well in one dataset but surprisingly poorly in another, \
	creating an illusion of progress. To address the issues above, we studied over one \
	hundred papers to identify, collect, process, and systematically \
	format datasets proposed in the past decades. \
	 \
	This webpage is grouping all the results and the analysis made"

footer_height = "6rem"
sidebar_width = "20rem"
table_height = "39rem"
table_height_2 = "59rem"
ts_height = "19rem"
scatter_height = "60rem"
paper_height = '60rem'
bck_height = "16rem" 
dataset_height = "80rem" 
table_height_2_1="42rem"

LIMIT_POINT_TS = 20000

PAPER_STYLE = {
    "margin-top": 0,
    "margin-left": 0,
    "margin-right": 0,
    "margin-bottom": 0,
    "maxHeight": paper_height,
    "background-color": "white",
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": footer_height,
    "width": sidebar_width,
    "padding": "1rem 1rem",
    "background-color": "lightgrey",
}


FOOTER_STYLE = {
    "position": "fixed",
    "bottom": 0,
    "left": 0,
    "right": 0,
    "height": footer_height,
    "padding": "1rem 1rem",
    "background-color": "rgb(55, 55, 55)",
}

CONTENT_STYLE = {
    "margin-top": 0,
    "margin-left": sidebar_width,
    "margin-right": 0,
    "margin-bottom": footer_height,
    "padding": "1rem 1rem",
}

CONTENT_STYLE_scatter = {
    "margin-top": 0,
    "margin-left": 0,
    "margin-right": 0,
    "margin-bottom": 0,
    "padding": "1rem 1rem",
    "maxHeight": scatter_height,
    "overflow": "scroll"
}

CONTENT_STYLE_bck = {
    "margin-top": 0,
    "margin-left": 0,
    "margin-right": 0,
    "margin-bottom": 0,
    "padding": "1rem 1rem",
    "maxHeight": bck_height,
    "overflow": "scroll"
}

CONTENT_STYLE_bck_dataset = {
    "margin-top": 0,
    "margin-left": 0,
    "margin-right": 0,
    "margin-bottom": 0,
    "padding": "1rem 1rem",
    "maxHeight": dataset_height,
    "overflow": "scroll"
}

CONTENT_STYLE_table_2={
    "margin-top": 0,
    "margin-left": 0,
    "margin-right": 0,
    "margin-bottom": 0,
    "padding": "1rem 1rem",
    "maxHeight": table_height_2,
    "overflow": "scroll"
}

CONTENT_STYLE_table_2_1={
    "margin-top": 0,
    "margin-left": 0,
    "margin-right": 0,
    "margin-bottom": 0,
    "padding": "1rem 1rem",
    "maxHeight": table_height_2_1,
    "overflow": "scroll"
}

CONTENT_STYLE_table = {
    "margin-top": 0,
    "margin-left": 0,
    "margin-right": 0,
    "margin-bottom": 0,
    "padding": "1rem 1rem",
    "maxHeight": table_height,
    "overflow": "scroll"
}

CONTENT_STYLE_ts = {
    "margin-top": 0,
    "margin-left": 0,
    "margin-right": 0,
    "margin-bottom": 0,
    "padding": "1rem 1rem",
}

CONTENT_STYLE_stat = {
    "margin-top": 0,
    "margin-left": sidebar_width,
    "margin-right": 0,
    "margin-bottom": footer_height,
    "padding": "1rem 1rem",
}

########################################################################
####################### BANNER, FOOTER #################################
########################################################################

footer = html.Div(
    children = [dbc.Row(
            children=[
                dbc.Col(html.Img( src='data:image/png;base64,{}'.format(encoded_image_pd.decode()),style={"width" : "210px"}), align="center"),
                dbc.Col(html.Img( src='data:image/png;base64,{}'.format(encoded_image_lp.decode()),style={"width" : "135px"}), align="center"),
                dbc.Col(html.Img( src='data:image/png;base64,{}'.format(encoded_image_dn.decode()),style={"width" : "120px"}), align="center"),
                dbc.Col(html.Img( src='data:image/png;base64,{}'.format(encoded_image_edf.decode()),style={"width" : "50px"}), align="center")

        ], className="text-center"),

        #dbc.Row( children=[dbc.Col(
        #                    html.P(["© 2021 copyright: ",html.A(" DinoLab",href="http://dino.mi.parisdescartes.fr/",className="text-warning")],
        #                           className="small my-2"),align="center"
        #                    )], className="text-center text-white bg-info"
        #        )

    ],
    style = FOOTER_STYLE
)



########################################################################
############################## Sidebar #################################
########################################################################

sidebar = html.Div(
		[
			html.H3("Anomaly Detection Benchmark", className="display-4"),
			html.Hr(),
			html.H5('A comparison of {} anomaly detection methods with {} accuracy measures on {} time series'.format(len(methods_key),len(measures_key),len(df))),	
			html.Hr(),
			html.H6('Paul Boniol, John Paparrizos, Themis Palpanas, Micheal J. Franklin'),	
			#html.P(
			#	description, className="lead",style={"maxHeight": "200px", "overflow": "scroll"},
			#),
			dbc.Nav(
				[
					dbc.NavLink("Home", href="/page-1", id="page-1-link"),
					dbc.NavLink("Overview", href="/page-2", id="page-2-link"),
					dbc.NavLink("Methods comparisons", href="/page-3", id="page-3-link"),
					dbc.NavLink("Measures comparisons", href="/page-4", id="page-4-link"),
					dbc.NavLink("Background", href="/page-5", id="page-5-link"),
					dbc.NavLink("References", href="/page-6", id="page-6-link"),
				],
				vertical=True,
				pills=True,
			),
		],
		style=SIDEBAR_STYLE,
		)




cdivs = [html.Div(id="page-content")]

content = html.Div(cdivs, style=CONTENT_STYLE)

########################################################
########################################################
#### 				LAYOUT				   		     ###
########################################################
########################################################


dataset_select_page_1 = dbc.Select(
    id="dataset_select_page_1",
    options=[{"label": "{}".format('ALL'), "value": "{}".format('ALL')}] +[
        {"label": "{}".format(name), "value": "{}".format(name)} for name in list(set(df['dataset'].values))
    ],value='ALL',
)

measure_select_page_1 = dbc.Select(
    id="measure_select_page_1",
    options=[
        {"label": "{}".format(name), "value": "{}".format(name)} for name in measures_key
    ],value='AUC_PR',
)

Type_anom_select_page_1 = dbc.Select(
    id="type_anom_select_page_1",
    options=[
        {"label": "ALL", "value": "ALL"},
        {"label": "Point anomaly", "value": "point"},
        {"label": "Sequence anomaly", "value": "sequence"},
    ],value='ALL',
)

Type_ts_select_page_1 = dbc.Select(
    id="type_ts_select_page_1",
    options=[
        {"label": "ALL", "value": "ALL"},
        {"label": "Single anomaly", "value": "single"},
        {"label": "Multiple anomalies", "value": "multiple"},
    ],value='ALL',
)








methodX_select_page_2 = dbc.Select(
    id="methodX_select_page_2",
    options=[{"label": "{}".format('ALL'), "value": "{}".format('ALL')}] +[
        {"label": "{}".format(name), "value": "{}".format(name)} for name in methods_key
    ],value='MP',
)

methodY_select_page_2 = dbc.Select(
    id="methodY_select_page_2",
    options=[{"label": "{}".format('ALL'), "value": "{}".format('ALL')}] +[
        {"label": "{}".format(name), "value": "{}".format(name)} for name in methods_key
    ],value='IFOREST',
)



dataset_select_page_2 = dbc.Select(
    id="dataset_select_page_2",
    options=[{"label": "{}".format('ALL'), "value": "{}".format('ALL')}] +[
        {"label": "{}".format(name), "value": "{}".format(name)} for name in list(set(df['dataset'].values))
    ],value='ALL',
)

measure_select_page_2 = dbc.Select(
    id="measure_select_page_2",
    options=[
        {"label": "{}".format(name), "value": "{}".format(name)} for name in measures_key
    ],value='AUC_PR',
)

Type_anom_select_page_2 = dbc.Select(
    id="type_anom_select_page_2",
    options=[
        {"label": "ALL", "value": "ALL"},
        {"label": "Point anomaly", "value": "point"},
        {"label": "Sequence anomaly", "value": "sequence"},
    ],value='ALL',
)

Type_ts_select_page_2 = dbc.Select(
    id="type_ts_select_page_2",
    options=[
        {"label": "ALL", "value": "ALL"},
        {"label": "Single anomaly", "value": "single"},
        {"label": "Multiple anomalies", "value": "multiple"},
    ],value='ALL',
)







dataset_select_page_3 = dbc.Select(
    id="dataset_select_page_3",
    options=[{"label": "{}".format('ALL'), "value": "{}".format('ALL')}] +[
        {"label": "{}".format(name), "value": "{}".format(name)} for name in list(set(df['dataset'].values))
    ],value='ALL',
)


exp_select_page_3 = dbc.Select(
    id="exp_select_page_3",
    options=[{"label": "Robustness to Noise", "value": "noise"},
        {"label": "Robustness to lag", "value": "lag"},
        {"label": "Robustness to Normal/Abnormal ratio", "value": "ratio"},
    ],
)


type_of_plot_3 = dbc.Select(
    id="type_plot_page_3",
    options=[{"label": "Boxplot", "value": "boxplot"},
        {"label": "Mean value", "value": "mean"},
        {"label": "Median value", "value": "median"},
        {"label": "Minimal value", "value": "min"},
        {"label": "Maximal value", "value": "max"}
    ],value='boxplot',
)


type_of_plot_3_1 = dbc.Select(
    id="type_plot_page_3_1",
    options=[{"label": "Boxplot", "value": "boxplot"},
        {"label": "evolution", "value": "evolution"},
        {"label": "Stanrdart Deviation", "value": "std"},
    ]
)

exp_select_page_3_1 = dbc.Select(
    id="exp_select_page_3_1",
    options=[{"label": "Robustness to Noise", "value": "noise"},
        {"label": "Robustness to lag", "value": "lag"},
        {"label": "Robustness to Normal/Abnormal ratio", "value": "ratio"},
    ],
)

time_series_select_page_3 = dcc.Dropdown(
    id="time_series_select_page_3",
    options=[{"label": "{}".format(name), "value": "{}".format(name)} for name in list(set(df['filename'].values))
    ],
)

method_select_page_3 = dbc.Select(
    id="method_select_page_3",
    options=[{"label": "{}".format(name), "value": "{}".format(name)} for name in methods_key_measures
    ],
)

condidtion_custom = dcc.Checklist(
    id="condition_custom_page_3",
    options=[{"label": "", "value": "custom"}]
)








