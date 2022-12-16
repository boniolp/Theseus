from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px

import pandas as pd
from dash import dash_table
#layout
from layout_tools import *
from background import *



modal_info_page_1 = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("More info")),
        dbc.ModalBody(
        	dcc.Markdown(convert(text_info_page_1),dangerously_allow_html=True,style={'text-align': 'justify'})
        ),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="close_1", outline=True, color="dark", n_clicks=0
            )
        ),
    ],
    id="modal_page_1",
    is_open=False,
    size="lg",
    )


modal_info_page_2 = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("More info")),
        dbc.ModalBody(
        	dcc.Markdown(convert(text_info_page_2),dangerously_allow_html=True,style={'text-align': 'justify'})
        ),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="close_2", outline=True, color="dark", n_clicks=0
            )
        ),
    ],
    id="modal_page_2",
    is_open=False,
    size="lg",
    )


modal_info_page_3 = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Accuracy Measures Evaluation")),
        dbc.ModalBody(
        	dcc.Markdown(convert(text_info_page_3),dangerously_allow_html=True,style={'text-align': 'justify'})
        ),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="close_3", outline=True, color="dark", n_clicks=0
            )
        ),
    ],
    id="modal_page_3",
    is_open=False,
    size="lg",
    )



def generate_page_1():
	return html.Div([
			dbc.Row([
				dbc.Col([
					html.H3("Anomaly Detection Benchmark", className="display-4"),
					html.Hr(),
					html.H5('A comparison of {} anomaly detection methods with {} accuracy measures on {} time series'.format(len(methods_key),len(measures_key),len(df))),	
					html.Hr(),
					html.Img( src='data:image/png;base64,{}'.format(encoded_image_intro.decode()),style={"width" : "500px",'text-align': 'center','margin': 'auto', 'display': 'block'}),
					html.Hr(),
					html.P(description)
				],
				align="center",
				width={"size": 6, "offset": 3},
				style={'text-align': 'justify'}),
			],
			align="center", className="h-50")
		], 
		style={'height':'100hv'}
	)



def generate_page_2(df_old,measures='AUC-ROC',dataset='ALL'):
	df_new = df_old[['filename']+methods_key]
	df_new = df_new.round(3)
	result_table = html.Div([dash_table.DataTable(df_new.to_dict('records'), [{"name": i, "id": i} for i in df_new.columns],id='accuracy_tbl')],id='div_table_page_1',#dbc.Table.from_dataframe(df[:200], striped=True, bordered=True, hover=True)],
		style=CONTENT_STYLE_table)
	result_ts = html.Div([html.P("")],id='ts_place',
		style=CONTENT_STYLE_ts)
	stat_ts = html.Div([html.P("")],id='stat_ts_place',
		style=CONTENT_STYLE_ts)
	
	to_plot = df_new[methods_key]
	fig = px.box(to_plot[to_plot.median().sort_values(ascending=True).index],labels={
                     "value": "{}".format(measures),
                     "variable": "{}".format('AD methods'),
                 },title="Average {} on {} time series".format(measures,dataset))
	fig.update_layout(template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=375)

	stat = dcc.Graph(figure=fig,id='boxplot_page_1')
	title_table = html.Div(children=[html.H5('{} for {} time series'.format(measures,len(df_new)))],id='title_table')
	return html.Div([
		dbc.Row([
			dbc.Col([html.H1('Overall Benchmark Evaluation')],width=10),
			dbc.Col([dbc.Button("More info", id="open_page_1", n_clicks=0,outline=True, color="dark"),modal_info_page_1,],width=2),
		]),
		html.Hr(),
		dbc.Row([
			dbc.Col([
				html.Div([
					dbc.Row([
						dbc.Col([
							html.P("Select dataset:"),
						],width=6),
						dbc.Col([
							dataset_select_page_1,
						],width=6),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select measure"),
						],width=6),
						dbc.Col([
							measure_select_page_1,
						],width=6),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select anomaly type:"),
						],width=6),
						dbc.Col([
							Type_anom_select_page_1,
						],width=6),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select time series Type:"),
						],width=6),
						dbc.Col([
							Type_ts_select_page_1,
						],width=6),
					]),
					html.Hr(),
					stat],style=CONTENT_STYLE_table),
				],width=3),
			dbc.Col([
				title_table,
				result_table]),
			]),
		html.Hr(),
		dbc.Row([
			dbc.Col([stat_ts],width=3),
			dbc.Col([result_ts],width=9),
			]
			)])
	#return html.P("This is the content of page 2!")





def generate_page_3(df_old):
	df_new = df_old[['filename']+methods_key]
	df_new = df_new.round(3)
	result_table = html.Div([html.P("")],id='comp_place',
		style=CONTENT_STYLE_scatter)
	result_ts = html.Div([dcc.Graph(figure = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16]),id='scatter_page_2',style={'display': 'none'})],id='ts_place_comp',
		style=CONTENT_STYLE_ts)
	stat_ts = html.Div([html.P("")],id='stat_ts_place_comp',
		style=CONTENT_STYLE_ts)
	stat_ts_all = html.Div([html.P("")],id='stat_ts_place_comp_all',
		style=CONTENT_STYLE_ts)
	
	
	return html.Div([
		dbc.Row([
			dbc.Col([html.H1('Anomaly Detection Methods comparison')],width=10),
			dbc.Col([dbc.Button("More info", id="open_page_2", n_clicks=0,outline=True, color="dark"),modal_info_page_2,],width=2),
		]),
		html.Hr(),
		dbc.Row([
			dbc.Col([
				html.Div([
					dbc.Row([
						dbc.Col([
							html.P("Method X:"),
							methodX_select_page_2,
						],width=6),
						dbc.Col([
							html.P("Method Y:"),
							methodY_select_page_2,
						],width=6),
					]),
					html.Hr(),
					dbc.Row([
						dbc.Col([
							html.P("Select dataset:"),
						],width=6),
						dbc.Col([
							dataset_select_page_2,
						],width=6),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select measure"),
						],width=6),
						dbc.Col([
							measure_select_page_2,
						],width=6),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select anomaly type:"),
						],width=6),
						dbc.Col([
							Type_anom_select_page_2,
						],width=6),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select Cardinality:"),
						],width=6),
						dbc.Col([
							Type_ts_select_page_2,
						],width=6),
					]),
					html.Hr(),
					stat_ts_all],style=CONTENT_STYLE_table_2),
				],width=3),
			dbc.Col([result_table],style=CONTENT_STYLE_table_2),
			]),
		html.Hr(),
		dbc.Row([
			dbc.Col([result_ts],width=12),
			]
			),
		])



def generate_page_4(df_old):
	df_new = df_old[['filename']+methods_key]
	df_new = df_new.round(3)
	title_table = html.Div(children=[html.H5('')],id='title_table_3')
	result_table = html.Div(children=[html.P('')],id='res_table_3')
	title_table_2 = html.Div(children=[html.H5('')],id='title_table_3_1')
	result_table_2 = html.Div(children=[html.P('')],id='res_table_3_1')
	result_ts = html.Div(children=[html.P('')],id='res_ts_3')
	return html.Div([
		dbc.Row([
			dbc.Col([html.H1('Accuracy measures Evaluation')],width=10),
			dbc.Col([dbc.Button("More info", id="open_page_3", n_clicks=0,outline=True, color="dark"),modal_info_page_3,],width=2),
		]),
		html.Hr(),
		#dbc.Row([
		dbc.Row([
			dbc.Col([
				html.Div([
					html.H5('Global Evaluation'),
					dbc.Row([
						dbc.Col([
							html.P("Select dataset:"),
						],width=4),
						dbc.Col([
							dataset_select_page_3,
						],width=8),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select experiment type:"),
						],width=4),
						dbc.Col([
							exp_select_page_3,
						],width=8),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select plot type:"),
						],width=4),
						dbc.Col([
							type_of_plot_3,
						],width=8),
					]),
					#html.Hr(),
					],style=CONTENT_STYLE_table_2_1),
				],width=3),
			dbc.Col([
				dbc.Row([dbc.Col([title_table,result_table],width=12)]),
				],width=9),],style=CONTENT_STYLE_table_2_1),
			html.Hr(),
		dbc.Row([
			dbc.Col([
				html.Div([
					#html.Hr(),
					dbc.Row([
						dbc.Col([
							condidtion_custom,
						],width=2),
						dbc.Col([
							html.H5('Create your own experiment'),
						],width=10),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select Time series:"),
						],width=4),
						dbc.Col([
							time_series_select_page_3,
						],width=8),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select AD methods:"),
						],width=4),
						dbc.Col([
							method_select_page_3,
						],width=8),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select experiment type:"),
						],width=4),
						dbc.Col([
							exp_select_page_3_1,
						],width=8),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select plot type:"),
						],width=4),
						dbc.Col([
							type_of_plot_3_1,
						],width=8),
					]),
					#html.Hr(),
					],style=CONTENT_STYLE_table_2),
				],width=3),

			dbc.Col([
				dbc.Row([dbc.Col([title_table_2,result_table_2],width=12)]),
				],width=9),],style=CONTENT_STYLE_table_2),


			#],style=CONTENT_STYLE_table_2),
			
		html.Hr(),
		dbc.Row([
			#html.Progress(id="progress_bar"),
			dbc.Col([result_ts],width=12),
			]
			)])




def generate_page_perso_dataset(df_old):

	title_table 	= html.Div(children=[html.H5('Accuracy evaluation on your own dataset')],id='title_table_perso_dataset')
	result_table 	= html.Div(children=[])
	stat_ts 		= html.Div(children=[])
	result_ts		= html.Div(children=[])
	return html.Div([
		dbc.Row([
			dbc.Col([html.H1('Evaluate on your data')],width=10),
			dbc.Col([dbc.Button("More info", id="open_page_perso_dataset", n_clicks=0,outline=True, color="dark"),modal_info_page_perso_dataset,],width=2),
		]),
		html.Hr(),
		dbc.Row([
			dbc.Col([
				html.Div([
					dbc.Row([
						dbc.Col([
							html.P("Upload your dataset:"),
						],width=6),
						#dbc.Col([
						#	dataset_select_page_perso,
						#],width=6),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select measure"),
						],width=6),
						#dbc.Col([
						#	measure_select_page_1,
						#],width=6),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select anomaly type:"),
						],width=6),
						#dbc.Col([
						#	Type_anom_select_page_1,
						#],width=6),
					]),
					dbc.Row([
						dbc.Col([
							html.P("Select time series Type:"),
						],width=6),
						#dbc.Col([
						#	Type_ts_select_page_1,
						#],width=6),
					]),
					html.Hr(),
					#stat,
				],style=CONTENT_STYLE_table),
			],width=3),
			dbc.Col([
				title_table,
				result_table]),
			]),
		html.Hr(),
		dbc.Row([
			dbc.Col([stat_ts],width=3),
			dbc.Col([result_ts],width=9),
			]
			)])
	#return html.P("This is the content of page 2!")





def generate_page_5():
	return html.Div([
		html.H1('Notations, Definitions and Methods'),
		html.Hr(),
		dbc.Row([
			dbc.Col([
					dcc.Markdown(convert(background_notation),dangerously_allow_html=True,style={'text-align': 'justify'})
				],width=4,style=CONTENT_STYLE_bck_dataset),
			dbc.Col([
				dcc.Markdown(text_background,style={'text-align': 'justify'}),
			],width=4,style=CONTENT_STYLE_bck_dataset),
			dbc.Col([
				dcc.Markdown(background_method,style={'text-align': 'justify'}),
				html.Hr(),
				dcc.Markdown(background_method_param,style={'text-align': 'justify'})
			],width=4,style=CONTENT_STYLE_bck_dataset),
		])
	])


def generate_page_6():
	return html.Div([
			#html.H2('Related Research Papers'),
			#html.Hr(),
			#dbc.Row([
			#	dbc.Col([
			#		html.H4('TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection'),
			#	],width=4),
			#	dbc.Col([
			#		html.H4('Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection'),
			#	],width=4),
			#	dbc.Col([
			#		html.H4('Demo benchmark: TO DEFINE'),
			#	],width=4),
			#]),
			#dbc.Row([
			#	dbc.Col([
			#		html.Hr(),
			#		html.H6('John Paparrizos, Yuhao Kang, Paul Boniol, Ruey S. Tsay, Themis Palpanas, Micheal J. Franklin'),
			#	],width=4),
			#	dbc.Col([
			#		html.Hr(),
			#		html.H6('John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay, Aarone Elmore, Micheal J. Franklin'),
			#	],width=4),
			#	dbc.Col([
			#		html.Hr(),
			#		html.H6('Paul Boniol, John Paparrizos, Themis Palpanas, Ruey S. Tsay, Aarone Elmore, Micheal J. Franklin'),
			#	],width=4),
			#]),
			#dbc.Row([
			#	dbc.Col([
			#		html.Hr(),
			#		html.P('Proceedings of the VLDB Endowement, Volume X, Issue X, 2022'),
			#	],width=4),
			#	dbc.Col([
			#		html.Hr(),
			#		html.P('Proceedings of the VLDB Endowement, Volume X, Issue X, 2022'),
			#	],width=4),
			#	dbc.Col([
			#		html.Hr(),
			#		html.P('Proceedings of the VLDB Endowement, Volume X, Issue X, 2022'),
			#	],width=4),
			#]),
			#dbc.Row([
			#	dbc.Col([
			#		html.Hr(),
			#		html.Iframe(id="embedded-pdf", src="assets/Benchmark.pdf#toolbar=0&navpanes=0&scrollbar=0",style={'width': '100%', 'height':'85%'}),
			#	],width=4),
			#	dbc.Col([
			#		html.Hr(),
			#		html.Iframe(id="embedded-pdf", src="assets/VUS.pdf#toolbar=0&navpanes=0&scrollbar=0",style={'width': '100%', 'height':'85%'}),
			#	],width=4),
			#	dbc.Col([
			#		html.Hr(),
			#		html.Iframe(id="embedded-pdf", src="assets/demopaper.pdf#toolbar=0&navpanes=0&scrollbar=0",style={'width': '100%', 'height':'85%'}),
			#	],width=4),
			#],className="h-50"),
			dbc.Row([
				dbc.Col([
					dcc.Markdown(references_text,style={'text-align': 'justify'})
				])
			])
		],style={"height": "100vh"})