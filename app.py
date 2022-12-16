import pathlib
import os
import io

import pandas as pd
import numpy as np
import base64
import json

import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State


from random import shuffle
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from tqdm import tqdm as tqdm
import time
from sklearn.preprocessing import MinMaxScaler
import random


import sys
from src.utils.slidingWindows import find_length
from src.utils.metrics import metricor
from src.models.distance import Fourier
from src.models.feature import Window
from src.analysis.score_computation import generate_data

import constants
#from navbar import sand_paper_navbar
from navbar import *
from layout_tools import *
from dash.long_callback import DiskcacheLongCallbackManager

external_stylesheets = [
	{
		'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
		'rel': 'stylesheet',
		'integrity': 'sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf',
		'crossorigin': 'anonymous'
	},dbc.themes.BOOTSTRAP
]

import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# app initialize
app = dash.Dash(
	__name__,
	meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
	external_stylesheets=external_stylesheets,long_callback_manager=long_callback_manager
)
app.css.append_css({'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'}) #fontawesome
server = app.server
app.config["suppress_callback_exceptions"] = True


#################################################################################
########################## ADD LOCAL CSS OR JAVASCRIPT ##########################
#################################################################################

css_directory = os.getcwd()
assets= "assets/"
locals = [assets+'loader.css']
static_css_route = '/static/'

@app.server.route('{}<stylesheet>'.format(static_css_route))
def serve_stylesheet(locals):
	if stylesheet not in locals:
		raise Exception(
			'"{}" is excluded from the allowed static files'.format(
				stylesheet
			)
		)
	return flask.send_from_directory(css_directory, stylesheet)

for local in locals:
	if ".css" in local:
		app.css.append_css({"external_url": "/static/{}".format(local)})
	if ".js" in local:
		app.scripts.append_script({'external_url': "/static/{}".format(local)})



########################################################################
############################## LAYOUT ##################################
########################################################################

base_layout = html.Div([
	dcc.Location(id="url"), 
	sidebar, 
	content,
	footer,
	],
)


app.layout = base_layout

########################################################################
############################## CALLBACK ################################
########################################################################

@app.callback(
	[Output(f"page-{i}-link", "active") for i in range(1, 7)],
	[Input("url", "pathname")],
)
def toggle_active_links(pathname):
	if pathname == "/":
		# Treat page 1 as the homepage / index
		return True, False, False, False, False, False
	return [pathname == f"/page-{i}" for i in range(1, 7)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
	if pathname in ["/", "/page-1"]:
		return generate_page_1()
	elif pathname == "/page-2":
		return generate_page_2(df)
	elif pathname == "/page-3":
		return generate_page_3(df)
	elif pathname == "/page-4":
		return generate_page_4(df)
	elif pathname == "/page-5":
		return generate_page_5()
	elif pathname == "/page-6":
		return generate_page_6()
		
	# If the user tries to reach a different page, return a 404 message
	return dbc.Jumbotron(
		[
			html.H1("404: Not found", className="text-danger"),
			html.Hr(),
			html.P(f"The pathname {pathname} was not recognised..."),
		]
	)



def toggle_modal(n1,n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


app.callback(
    Output("modal_page_1", "is_open"),
    [Input("open_page_1", "n_clicks"), Input("close_1", "n_clicks")],
    State("modal_page_1", "is_open"),
)(toggle_modal)

app.callback(
    Output("modal_page_2", "is_open"),
    [Input("open_page_2", "n_clicks"), Input("close_2", "n_clicks")],
    State("modal_page_2", "is_open"),
)(toggle_modal)

app.callback(
    Output("modal_page_3", "is_open"),
    [Input("open_page_3", "n_clicks"), Input("close_3", "n_clicks")],
    State("modal_page_3", "is_open"),
)(toggle_modal)


############################## page 1 ################################

def add_rect(label,data):
	anom_plt = [None]*len(data)
	ts_plt = data.copy()
	len_ts = len(data)
	for i,lab in enumerate(label):
		if lab == 1:
			anom_plt[i] = data[i]
			anom_plt[min(len_ts-1,i+1)] = data[min(len_ts-1,i+1)]
	return anom_plt


@app.callback(
	[Output('stat_ts_place', 'children'),
	Output('ts_place', 'children')], 
	[Input('accuracy_tbl', 'active_cell'),Input('accuracy_tbl', 'data')])
def update_graphs(active_cell,data_cell):
	if active_cell:
		folder = df.loc[df['filename']==data_cell[active_cell['row']]['filename']]['dataset'].values[0]
		if 'NASA_' in folder:
			folder = folder.replace('NASA_','NASA-')
			path = path_top_dataseries + folder + '/' + data_cell[active_cell['row']]['filename'].replace("SMAP", "").replace('_data.out','.test.out')
			path_anom = path_top_anoamly_score + folder + '/{}/score/' + data_cell[active_cell['row']]['filename'].replace("SMAP", "").replace('_data.out','.test.out')
		else:
			path = path_top_dataseries + folder + '/' + data_cell[active_cell['row']]['filename'].replace(".txt", ".out")
			path_anom = path_top_anoamly_score + folder + '/{}/score/' + data_cell[active_cell['row']]['filename'].replace(".txt", ".out")
		ts = pd.read_csv(path + '.zip',compression='zip', header=None).to_numpy()
		
		



		

		label = ts[:,1]
		data = ts[:,0].astype(float)

		scores = {}
		for method_name in os.listdir(path_top_anoamly_score + folder + '/'):
			if (method_name in methods_key) and (os.path.isfile(path_anom.format(method_name)+ '.zip')):
				print('found {}'.format(method_name))
				scores_tmp = pd.read_csv(path_anom.format(method_name)+ '.zip',compression='zip', header=None).to_numpy()
				scores[method_name] = scores_tmp[:,0].astype(float)
		
		

		#fig = px.line(data)
		anom = add_rect(label,data)
		trace_scores = []
		trace_scores.append(go.Scattergl(
			x=list(range(len(data))),
			y=data,
			xaxis='x',
			yaxis='y2',
			name = "Time series",
			mode = 'lines',
			line = dict(color = 'blue',width=3),
			opacity = 1
		))
		trace_scores.append(go.Scattergl(
			x=list(range(len(data))),
			y=anom,
			xaxis='x',
			yaxis='y2',
			name = "Anomalies",
			mode = 'lines',
			line = dict(color = 'red',width=3),
			opacity = 1
		))

		for method_name in scores.keys():
			trace_scores.append(go.Scattergl(
				x=list(range(len(data))),
				y=[0] + list(scores[method_name][1:-1]) + [0],
				name = "{} score".format(method_name),
				opacity = 1,
				mode = 'lines',
				fill="tozeroy",
			))



		layout = go.Layout(
			yaxis=dict(
				domain=[0, 0.4],
				range=[0,1]
			),
			yaxis2=dict(
				domain=[0.45, 1],
				range=[min(data),max(data)]
			),
			#showlegend=False,
			title="{} time series snippet (40k points maximum)".format(data_cell[active_cell['row']]['filename'].split(".")[0]),
			template="simple_white",
			margin=dict(l=8, r=4, t=50, b=10),
			height=375,
			hovermode="x unified",
			xaxis=dict(
				range=[0,len(data)]
			)
		)

		fig = dict(data=trace_scores, layout=layout)

		#fig.update_layout(hoverdistance=1)
		to_plot = df.loc[df['filename']==data_cell[active_cell['row']]['filename']][methods_key].mean()

		fig_bar = px.bar(to_plot,labels={
					 "value": "{}".format('Accuracy'),
					 "index": "{}".format('AD methods'),
				 },title="{} on {} time series".format('Accuracy',data_cell[active_cell['row']]['filename'].split(".")[0]))
		fig_bar.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=375)


		return [dcc.Graph(figure=fig_bar,id='stat_ts_place_1',style={'width': '100%'})],[dcc.Graph(figure=fig,id='ts_place_1',style={'width': '100%'})]
	return None,None

@app.callback(
	[Output('title_table','children'),
	Output('boxplot_page_1','figure'),
	Output('div_table_page_1', 'children')], 
	[Input('dataset_select_page_1', 'value'),
	Input('measure_select_page_1', 'value'),
	Input('type_anom_select_page_1', 'value'),
	Input('type_ts_select_page_1', 'value'),])
def update_graphs(dataset,measure,anoma_type,ts_type):
	global df
	df_new = df
	
	if measure is not None:
		df_new = pd.read_csv('data/mergedTable_{}.csv'.format(measure))

	if dataset == 'ALL':
		df_new = df_new
	elif dataset is not None:
		df_new = df_new.loc[df_new['dataset'] == dataset]

	if anoma_type == 'ALL':
		df_new = df_new
	elif anoma_type is not None:
		df_new = df_new.loc[df_new['type_an'] == anoma_type]

	if ts_type == 'ALL':
		df_new = df_new
	elif ts_type == 'single':
		df_new = df_new.loc[df_new['nb_anomaly'] == 1.0]
	elif ts_type == 'multiple':
		df_new = df_new.loc[df_new['nb_anomaly'] > 1.0]
	df_new = df_new[['filename']+methods_key]
	df_new = df_new.round(3)


	if dataset is None: dataset = 'ALL'
	if measure is None: measure = 'AUC-ROC'
	if anoma_type is None: anoma_type = 'ALL'
	to_plot = df_new[methods_key] 
	fig = px.box(to_plot[to_plot.median().sort_values(ascending=True).index],labels={
					 "value": "{}".format(measure),
					 "variable": "{}".format('AD methods'),
				 },title="Average {} on {} time series ({})".format(measure,dataset,anoma_type))
	fig.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=375)
	

	return html.H5('{} for {} time series'.format(measure,len(df_new))),fig,[dash_table.DataTable(df_new.to_dict('records'), [{"name": i, "id": i} for i in df_new.columns],id='accuracy_tbl')]


############################## page 2 ################################

@app.callback(
	[Output('stat_ts_place_comp_all','children'),
	Output('comp_place', 'children')], 
	[Input('methodX_select_page_2', 'value'),
	Input('methodY_select_page_2', 'value'),
	Input('dataset_select_page_2', 'value'),
	Input('measure_select_page_2', 'value'),
	Input('type_anom_select_page_2', 'value'),
	Input('type_ts_select_page_2', 'value'),])
def update_comp(methodX,methodY,dataset,measure,anoma_type,ts_type):
	global df
	df_new = df

	if measure is not None:
		df_new = pd.read_csv('data/mergedTable_{}.csv'.format(measure))

	if (methodX is not None) and (methodY is not None):
		if dataset == 'ALL':
			df_new = df_new
		elif dataset is not None:
			df_new = df_new.loc[df_new['dataset'] == dataset]

		if anoma_type == 'ALL':
			df_new = df_new
		elif anoma_type is not None:
			df_new = df_new.loc[df_new['type_an'] == anoma_type]

		if ts_type == 'ALL':
			df_new = df_new
		elif ts_type == 'single':
			df_new = df_new.loc[df_new['nb_anomaly'] == 1.0]
		elif ts_type == 'multiple':
			df_new = df_new.loc[df_new['nb_anomaly'] > 1.0]
		df_new = df_new[['filename','dataset']+methods_key]
		#df_new = df_new.round(3)


		if dataset is None: dataset = 'ALL'
		if measure is None: measure = 'AUC-ROC'
		if anoma_type is None: anoma_type = 'ALL'
		to_plot = df_new[[methodX,methodY,'dataset','filename']] 
		fig = px.box(to_plot[[methodX,methodY]],labels={
						 "value": "{}".format(measure),
						 "variable": "{}".format('methods'),
					 },title="Average {} on {} time series ({})".format(measure,dataset,anoma_type))
		fig.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=375)
		
		fig_scatter = px.scatter(to_plot,x=methodX, y=methodY,color='dataset',hover_name='filename',marginal_x='histogram', marginal_y='histogram')
		fig_scatter.update_traces(
			marker=dict(size=8,
			line=dict(width=1,
			color='DarkSlateGrey')),
			selector=dict(mode='markers'))
		#fig_scatter.add_trace(go.Scatter(x=to_plot[methodX], y=to_plot[methodY],
		#	mode='markers',name='markers')
		#)
		fig_scatter.add_trace(go.Scatter(x=[0,1], y=[0,1],
			mode='lines',name='equality lines',line=dict(width=2,color='black'))
		)
		
		fig_scatter.update_yaxes(rangemode="tozero")
		fig_scatter.update_xaxes(rangemode="tozero")
		fig_scatter.update_layout(template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=675)
		return [dcc.Graph(figure=fig,id='boxplot_page_2')],[dcc.Graph(figure=fig_scatter,id='scatter_page_2')]
	return None,None


@app.callback(Output('ts_place_comp','children'),
	[Input('scatter_page_2','clickData'),
	Input('methodX_select_page_2', 'value'),
	Input('methodY_select_page_2', 'value'),])
def display_hover_point(clickData,methodX,methodY):
	if (methodX is not None) and (methodY is not None):
		if (clickData is not None):
			folder = df.loc[df['filename']==clickData['points'][0]["hovertext"]]['dataset'].values[0]
			
			if 'NASA_' in folder:
				folder = folder.replace('NASA_','NASA-')
				path = path_top_dataseries + folder + '/' + clickData['points'][0]["hovertext"].replace("SMAP", "").replace('_data.out','.test.out')
				path_anom = path_top_anoamly_score + folder + '/{}/score/' + clickData['points'][0]["hovertext"].replace("SMAP", "").replace('_data.out','.test.out')
			else:
				path = path_top_dataseries + folder + '/' + clickData['points'][0]["hovertext"].replace(".txt", ".out")
				path_anom = path_top_anoamly_score + folder + '/{}/score/' + clickData['points'][0]["hovertext"].replace(".txt", ".out")
			ts = pd.read_csv(path+ '.zip',compression='zip', header=None).to_numpy()

			
			label = ts[:,1]
			data = ts[:,0].astype(float)
			fig = go.Figure()
			
			scores = {}
			for method_name in os.listdir(path_top_anoamly_score + folder + '/'):
				if (method_name in [methodX,methodY]) and (os.path.isfile(path_anom.format(method_name)+ '.zip')):
					print('found {}'.format(method_name))
					scores_tmp = pd.read_csv(path_anom.format(method_name)+ '.zip',compression='zip', header=None).to_numpy()
					scores[method_name] = scores_tmp[:,0].astype(float)
			
			

			#fig = px.line(data)
			anom = add_rect(label,data)
			trace_scores = []
			trace_scores.append(go.Scattergl(
				x=list(range(len(data))),
				y=data,
				xaxis='x',
				yaxis='y2',
				name = "Time series",
				mode = 'lines',
				line = dict(color = 'blue',width=3),
				opacity = 1
			))
			trace_scores.append(go.Scattergl(
				x=list(range(len(data))),
				y=anom,
				xaxis='x',
				yaxis='y2',
				name = "Anomalies",
				mode = 'lines',
				line = dict(color = 'red',width=3),
				opacity = 1
			))

			for method_name in scores.keys():
				trace_scores.append(go.Scattergl(
					x=list(range(len(data))),
					y=[0] + list(scores[method_name][1:-1]) + [0],
					name = "{} score".format(method_name),
					opacity = 1,
					mode = 'lines',
					fill="tozeroy",
				))



			layout = go.Layout(
				yaxis=dict(
					domain=[0, 0.4],
					range=[0,1]
				),
				yaxis2=dict(
					domain=[0.45, 1],
					range=[min(data),max(data)]
				),
				#showlegend=False,
				title="{} time series snippet (40k points maximum)".format(clickData['points'][0]["hovertext"].split(".")[0]),
				template="simple_white",
				margin=dict(l=8, r=4, t=50, b=10),
				height=375,
				hovermode="x unified",
				xaxis=dict(
					range=[0,len(data)]
				)
			)

			fig = dict(data=trace_scores, layout=layout)
			
			return [dcc.Graph(figure=fig,id='ts_place_2',style={'width': '100%'})]



############################## page 3 ################################

@app.callback(
	[Output('title_table_3','children'),
	Output('res_table_3','children')], 
	[Input('dataset_select_page_3', 'value'),
	Input('exp_select_page_3', 'value'),
	Input('type_plot_page_3', 'value')
	])
def update_graphs(dataset,exp,plot_type):
	global global_dataframe
	df_new = global_dataframe
	if dataset == 'ALL':
		df_new = df_new
	elif dataset is not None:
		df_new = df_new.loc[df_new['folder'] == dataset]

	if exp == 'noise':
		df_new = df_new.loc[df_new['type'] == 'noise']
	elif exp == 'lag':
		df_new = df_new.loc[df_new['type'] == 'lag']
	elif exp == 'ratio':
		df_new = df_new.loc[df_new['type'] == 'ratio']


	if exp is None:
		exp = 'lag,noise, and ratio'
	if dataset is None:
		dataset = 'ALL'
	#df_new = df_new[['filename']+methods_key]
	df_new = df_new.round(3)


	to_plot = df_new
	
	if plot_type is None:
		plot_type = 'boxplot'

	if plot_type == 'boxplot':
		fig = px.box(to_plot,y="value",x="measure",labels={
			"value": "{}".format("standard deviation"),
			"measure": "{}".format('Accuracy measures'),
		})
	elif plot_type == 'mean':
		fig = px.bar(to_plot[['measure','value']].groupby('measure').mean().sort_values('value',ascending=False),labels={
			"_value": "{}".format("average standard deviation"),
			"measure": "{}".format('Accuracy measures'),
		})
	elif plot_type == 'median':
		fig = px.bar(to_plot[['measure','value']].groupby('measure').median().sort_values('value',ascending=False),labels={
			"_value": "{}".format("median standard deviation"),
			"measure": "{}".format('Accuracy measures'),
		})
	elif plot_type == 'min':
		fig = px.bar(to_plot[['measure','value']].groupby('measure').min().sort_values('value',ascending=False),labels={
			"_value": "{}".format("minimal standard deviation"),
			"measure": "{}".format('Accuracy measures'),
		})
	elif plot_type == 'max':
		fig = px.bar(to_plot[['measure','value']].groupby('measure').max().sort_values('value',ascending=False),labels={
			"_value": "{}".format("maximal standard deviation"),
			"measure": "{}".format('Accuracy measures'),
		})

	fig.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=300)
	

	return html.H5("standard deviation when we inject {} in the anomaly score on {} time series".format(exp,dataset)),[dcc.Graph(figure=fig,id='box_place_3',style={'width': '100%'})]


def generate_new_label(label,lag):
	if lag < 0:
		return np.array(list(label[-lag:]) + [0]*(-lag))
	elif lag > 0:
		return np.array([0]*lag + list(label[:-lag]))
	elif lag == 0:
		return label

def generate_curve(label,score,slidingWindow):
	tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume(labels_original=label, score=score, windowSize=1*slidingWindow)

	#X = np.array(tpr_3d).reshape(1,-1).ravel()
	#X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
	#Y = np.array(fpr_3d).reshape(1,-1).ravel()
	#W = np.array(prec_3d).reshape(1,-1).ravel()
	#Z = np.repeat(window_3d, len(tpr_3d[0]))
	#Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)
	
	return avg_auc_3d, avg_ap_3d

@app.callback(
	output=[
		Output('title_table_3_1','children'),
		Output('res_table_3_1','children'),
		Output('res_ts_3','children')
	], 
	inputs=[
		Input('time_series_select_page_3', 'value'),
		Input('exp_select_page_3_1', 'value'),
		Input('type_plot_page_3_1', 'value'),
		Input('method_select_page_3', 'value'),
		Input('condition_custom_page_3', 'value')
	],)
	#progress=[
	#	Output("progress_bar", "value"), 
	#	Output("progress_bar", "max")
	#],prevent_initial_call=True)
def update_graphs_page_measure(time_series,exp,plot_type,method,condition_custom):
	#global global_dataframe
	global df


	if (condition_custom is not None) and (time_series is not None) and (method is not None) and (exp is not None) and (plot_type is not None):
		##### get data

		folder = df.loc[df['filename']==time_series]['dataset'].values[0]

		if 'NASA_' in folder:
			folder = folder.replace('NASA_','NASA-')
			path = path_top_dataseries + folder + '/' + time_series.replace("SMAP", "").replace('_data.out','.test.out')
			path_anom = path_top_anoamly_score + folder + '/{}/score/' + time_series.replace("SMAP", "").replace('_data.out','.test.out')
		else:
			path = path_top_dataseries + folder + '/' + time_series.replace(".txt", ".out")
			path_anom = path_top_anoamly_score + folder + '/{}/score/' + time_series.replace(".txt", ".out")
		
		ts = pd.read_csv(path+ '.zip',compression='zip', header=None).to_numpy()

		

		label = ts[:,1]
		data = ts[:,0].astype(float)
		fig = go.Figure()
		
		scores = pd.read_csv(path_anom.format(method)+ '.zip',compression='zip', header=None).to_numpy()
		scores = scores[:,0].astype(float)
		
		##### compute Exp

		pos_first_anom,slidingWindow,_,_,_,_,_,_,_ = generate_data(path+ '.zip',0,max_length=10000)

		dict_acc = {
				'R_AUC_ROC':      {},
				'AUC_ROC':        {},
				'R_AUC_PR':       {},
				'AUC_PR':         {},
				'VUS_ROC':        {},
				'VUS_PR':         {},
				'Precision':      {},
				'Recall':         {},
				'F':              {},
				'Precision@k':    {},
				'Rprecision':     {},
				'Rrecall':        {},
				'RF':             {}}

		if exp == 'lag':
			lag_range = list(range(-slidingWindow//4,slidingWindow//4,5))
		elif exp == 'noise':	
			lag_range = [0.01,0.02,0.05,0.07,0.1,0.12,0.15,0.17,0.2]

		for iter_lag,lag in enumerate(lag_range):
			print(iter_lag)
		
			if exp == 'lag':
				new_label = generate_new_label(label,lag)
				new_scores = scores
			elif exp == 'noise':
				new_label = label
				noise = np.random.normal(-lag,lag,len(scores))
				new_scores = np.array(scores) + noise
				new_scores = (new_scores - min(new_scores))/(max(new_scores) - min(new_scores))

			grader = metricor()  

			R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=new_label, score=new_scores, window=slidingWindow, plot_ROC=True) 
			L, fpr, tpr= grader.metric_new(new_label, new_scores, plot_ROC=True)
			precision, recall, AP = grader.metric_PR(new_label, new_scores)
			avg_auc_3d, avg_ap_3d = generate_curve(new_label,new_scores,2*slidingWindow)
			L1 = [ elem for elem in L]

			dict_acc['R_AUC_ROC'][lag] 		=R_AUC
			dict_acc['AUC_ROC'][lag]        =L1[0]
			dict_acc['R_AUC_PR'][lag]       =R_AP
			dict_acc['AUC_PR'][lag]         =AP
			dict_acc['VUS_ROC'][lag]        =avg_auc_3d
			dict_acc['VUS_PR'][lag]         =avg_ap_3d
			dict_acc['Precision'][lag]      =L1[1]
			dict_acc['Recall'][lag]         =L1[2]
			dict_acc['F'][lag]              =L1[3]
			dict_acc['Precision@k'][lag]    =L1[9]
			dict_acc['Rprecision'][lag]     =L1[7]
			dict_acc['Rrecall'][lag]        =L1[4]
			dict_acc['RF'][lag]             =L1[8]

				#set_progress((str(iter_lag + 1), str(len(lag_range))))

			

		#elif exp == 'noise':

		#elif exp == 'ratio':

	
		##### stat plot

		dict_acc_df = pd.DataFrame(dict_acc)[pd.DataFrame(dict_acc).std().sort_values(ascending=False).index]
		#if plot_type == 'boxplot':
		fig_box = px.box(dict_acc_df,labels={
			"value": "{}".format("value"),"variable": "{}".format('Accuracy measures')})
		fig_box.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=230)
		#elif plot_type == 'evolution':
		fig_box_evo = px.line(dict_acc_df,markers=True,labels={
			"value": "{}".format("value"),"index": "{} injected".format(exp)})
		fig_box_evo.update_layout(showlegend=True,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=460,hovermode="x unified")
		#elif plot_type == 'std':
		fig_bar = px.bar(dict_acc_df.std(),labels={
			"value": "{}".format("standard deviation"),"index": "{}".format('Accuracy measures')})
		fig_bar.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=230,)
		


		col_box = dbc.Row([
			dbc.Col([
					dbc.Row([
						dbc.Col([
							dcc.Graph(figure=fig_box,id='box_plot',style={'width': '100%'},config={'displayModeBar': False})
							],width=12),
					]),
					dbc.Row([
						dbc.Col([
							dcc.Graph(figure=fig_bar,id='bar_plot',style={'width': '100%'},config={'displayModeBar': False})
						],width=12),
					]),
				],width=6),
			dbc.Col(
				[dcc.Graph(figure=fig_box_evo,id='box_plot_evo',style={'width': '100%'},config={'displayModeBar': False})],width=6)
			])

		##### ts

		anom = add_rect(label,data)
		trace_scores = []
		trace_scores.append(go.Scattergl(
			x=list(range(len(data))),
			y=data,
			xaxis='x',
			yaxis='y2',
			name = "Time series",
			mode = 'lines',
			line = dict(color = 'blue',width=3),
			opacity = 1
		))
		trace_scores.append(go.Scattergl(
			x=list(range(len(data))),
			y=anom,
			xaxis='x',
			yaxis='y2',
			name = "Anomalies",
			mode = 'lines',
			line = dict(color = 'red',width=3),
			opacity = 1
		))

		
		if exp == 'lag':
			for i,lag in enumerate(lag_range):
				if (i == 0) or (i == len(lag_range)-1):
					trace_scores.append(go.Scattergl(
						x=list(range(len(data))),
						y=[0]*abs(min(lag,0))+ [0] + list(scores[1+max(lag,0):-1-abs(min(lag,0))]) + [0] + [0]*max(lag,0),
						name = "{} with {} lag".format(method,lag),
						#opacity = 1,
						line = dict(color = 'black'),
						mode = 'lines',
						fill="tozeroy",
						fillcolor='rgba(26,150,65,0.1)'
					))
				else:
					trace_scores.append(go.Scattergl(
						x=list(range(len(data))),
						y=[0]*abs(min(lag,0))+ [0] + list(scores[1+max(lag,0):-1-abs(min(lag,0))]) + [0] + [0]*max(lag,0),
						name = "{} with {} lag".format(method,lag),
						#opacity = 1,
						line = dict(color = 'rgba(26,150,65,0.1)'),
						mode = 'lines',
						fill="tozeroy",
						fillcolor='rgba(26,150,65,0.1)'
					))
		elif exp == 'noise':
			for i,lag in enumerate(lag_range):
				noise = np.random.normal(-lag,lag,len(scores))
				new_scores = np.array(scores) + noise
				new_scores = (new_scores - min(new_scores))/(max(new_scores) - min(new_scores))
				if (i == 0) or (i == len(lag_range)-1):
					trace_scores.append(go.Scattergl(
						x=list(range(len(data))),
						y=[0] + list(new_scores[1:-1]) + [0],
						name = "{} with {} lag".format(method,lag),
						#opacity = 1,
						line = dict(color = 'black'),
						mode = 'lines',
						fill="tozeroy",
						fillcolor='rgba(26,150,65,0.1)'
					))
				else:
					trace_scores.append(go.Scattergl(
						x=list(range(len(data))),
						y=[0] + list(new_scores[1:-1]) + [0],
						name = "{} with {} lag".format(method,lag),
						#opacity = 1,
						line = dict(color = 'rgba(26,150,65,0.1)'),
						mode = 'lines',
						fill="tozeroy",
						fillcolor='rgba(26,150,65,0.1)'
					))

		layout = go.Layout(
			yaxis=dict(
				domain=[0, 0.4],
				range=[0,1]
			),
			yaxis2=dict(
				domain=[0.45, 1],
				range=[min(data),max(data)]
			),
			#showlegend=False,
			title="{} time series snippet (40k points maximum)".format(time_series),
			template="simple_white",
			margin=dict(l=8, r=4, t=50, b=10),
			height=375,
			hovermode="x unified",
			xaxis=dict(
				range=[0,len(data)]
			)
		)

		fig = dict(data=trace_scores, layout=layout)

		ts_plot = [dcc.Graph(figure=fig,id='ts_place_3',style={'width': '100%','height':'80%'})]

		##### title	
		title = html.H5("{} Experiment on {} time series with {} method".format(exp,time_series,method))


		return title,col_box, ts_plot
	return None,None,None

########################################################################
########################################################################
########################################################################

# Running the server
if __name__ == "__main__":
	app.run_server(debug=True)
