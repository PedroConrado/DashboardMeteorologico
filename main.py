import base64
import io

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

import dash
from dash import dash_table, dcc, html, Input, Output, State
from dash.dependencies import MATCH
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go

import pmdarima as pm


# Inicializando o Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

APP_MODE = 'exploratory'

MODEL_FOR_VARIABLE = {
    'PRECIPITAÇÃO TOTAL': pm.ARIMA(order=(0,0,1)),
    'PRESSÃO ATMOSFERICA': pm.ARIMA(order=(1,0,1), seasonal_order=(2,0,2,12)),
    'PRESSÃO ATMOSFERICA MAX.': pm.ARIMA(order=(1,0,1), seasonal_order=(2,0,2,12)),
    'PRESSÃO ATMOSFERICA MIN.': pm.ARIMA(order=(1,0,1), seasonal_order=(2,0,2,12)),
    'TEMPERATURA DO AR': pm.ARIMA(order=(1,0,0), seasonal_order=(2,0,2,12)),
    'TEMPERATURA PT. ORVALHO': pm.ARIMA(order=(0,0,0)),
    'TEMPERATURA ORVALHO MAX.': pm.ARIMA(order=(0,0,0)),
    'TEMPERATURA ORVALHO MIN.': pm.ARIMA(order=(0,0,0)),
    'DIREÇÃO DO VENTO': pm.ARIMA(order=(0,0,0)),
    'RAJADA DO VENTO': pm.ARIMA(order=(0,0,0)),
    'VELOCIDADE DO VENTO': pm.ARIMA(order=(0,0,0)),
    'TEMPERATURA MEDIA': pm.ARIMA(order=(1,0,1), seasonal_order=(1,0,1,24)),
    'TEMPERATURA MÍNIMA': pm.ARIMA(order=(1,0,1), seasonal_order=(1,0,1,24)),
    'TEMPERATURA MÁXIMA': pm.ARIMA(order=(1,0,1), seasonal_order=(1,0,1,24)),
    'UMIDADE RELATIVA DO AR': pm.ARIMA(order=(1,0,0), seasonal_order=(1,0,1,24)),
    'UMIDADE REL. MAX.': pm.ARIMA(order=(1,0,0), seasonal_order=(1,0,1,24)),
    'UMIDADE REL. MIN.': pm.ARIMA(order=(1,0,0), seasonal_order=(1,0,1,24)),
    'RADIAÇÃO GLOBAL': pm.ARIMA(order=(1,0,0), seasonal_order=(1,0,1,12))
}


# Estilos customizados
CUSTOM_STYLE = {
    'background-color': '#f0f0f0',
    'padding': '10px',
    'border-radius': '10px',
    'box-shadow': '0px 2px 10px rgba(0, 0, 0, 0.1)'
}

TITLE_STYLE = {
    'font-weight': 'bold',
    'color': '#2c3e50',
    'font-size': '20px',
    'text-align': 'center',
    'margin-bottom': '10px'
}

SELECTION_TITLE_STYLE = {
    'font-weight': 'bold',
    'color': '#2980b9',
    'font-size': '16px',
    'text-align': 'center',
}

# Layout do app
app.layout = html.Div([
    # Título principal
    html.H1("Dashboard Meteorológico", style={'text-align': 'center', 'margin-top': '20px'}),
    
    # Componente de upload de CSV
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Arraste e solte ou ', html.A('selecione um arquivo')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    
    # Exibição dos dados carregados
    dcc.Loading(
        id="loading-table",
        type="circle",
        children=[
            html.Div(id='table-container', style={'margin': '20px'}),
        ],
    ),

    
    # Painel horizontal para seleções
    html.Div([
        # Seleção de range de data
        html.Div([
            html.P("Seleção de Datas", style=SELECTION_TITLE_STYLE),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date_placeholder_text="Data inicial",
                end_date_placeholder_text="Data final"
            )
        ], style={'flex': '1', 'padding': '10px', 'text-align': 'center'}),
        
        # Seleção de variáveis
        html.Div([
            html.P("Seleção de Variáveis", style=SELECTION_TITLE_STYLE),
            dcc.Dropdown(id='variable-dropdown', placeholder="Selecione variáveis", multi=True),
        ], style={'flex': '1', 'padding': '10px'}),
        
        # Seleção do tipo de análise (marcar o botão ativo)
        html.Div([
            html.P("Seleção do Tipo de Análise", style=SELECTION_TITLE_STYLE),
            dbc.ButtonGroup(
                [
                    dbc.Button("Análise Exploratória", id="exploratory-btn", n_clicks=0, color="primary", outline=False, active=True),
                    dbc.Button("Séries Temporais", id="time-series-btn", n_clicks=0, color="primary", outline=False, active=False)
                ],
                className="d-flex justify-content-center"
            )
        ], style={'flex': '1', 'padding': '10px'})
    ], style={**CUSTOM_STYLE, 'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

    # Botão de informações técnicas
    html.Div([
        html.A(html.Button('Documentação e Código', id='technical-info-button', style={'background-color': '#2980b9', 'color': 'white'}),
               href="https://github.com/PedroConrado/DashboardMeteorologico", target="_blank"),
    ], style={'text-align': 'center', 'margin': '20px'}),

    html.Div(
        id='model-analysis-container',
        children=[
            dcc.Checklist(
                id='model-analysis-checkbox',
                options=[
                    {'label': ' Exibir Análises de Modelagem', 'value': 'enabled'}
                ],
                value=[],  # Desabilitado por padrão
                style={'font-size': '16px'}
            )
        ],
        style={'display': 'none', 'text-align': 'right', 'margin': '20px'}  # Escondido inicialmente
    ),
    
    # Exibição dos gráficos
    html.Div(
        id='content-container',
        children=[
            dcc.Graph(id='correlation-matrix', style={'display': 'none'}),
            html.Div(id='time-series-plot-container', style={'display': 'none'}),
            dcc.Graph(id='time-series-graph', style={'display': 'none'}),
            

            # Gráfico de comparação por estação
            html.Div([
                dcc.Graph(id='variable-station-comparison', style={'display': 'none'}),
                html.P("", 
                    style={'text-align': 'center', 'font-size': '12px', 'color': '#7f8c8d', 'margin-top': '10px'})
            ]),

            # Gráficos extras para análise exploratória
            dcc.Graph(id='scatter-plot', style={'display': 'none'}),
            html.Div(id='histogram-plot-container', style={'display': 'none'})
        ],
        style={'display': 'none'}
    ),

    html.Div(
        id='content-loading-text-container',
        children=[],
        style={'display': 'none'}
    )
])

# Função para parsear o CSV
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

# Callback para esconder ou mostrar checkbox de análises de modelagem
@app.callback(
    Output('model-analysis-container', 'style'),
    [Input('exploratory-btn', 'n_clicks'),
     Input('time-series-btn', 'n_clicks')]
)
def toggle_checkbox_visibility(exploratory_clicks, time_series_clicks):
    # Identificar qual botão foi clicado
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Mostrar o checkbox somente na visualização de séries temporais
    if button_id == 'time-series-btn' or (time_series_clicks > exploratory_clicks):
        return {'display': 'block', 'text-align': 'right', 'margin': '20px'}  # Mostrar checkbox, alinhado à direita
    return {'display': 'none'}  # Esconder checkbox


# Callback para carregar e exibir a tabela dos dados e gerar opções de variáveis
@app.callback(
    [Output('table-container', 'children'),
     Output('variable-dropdown', 'options'),
     Output('date-picker-range', 'start_date'),
     Output('date-picker-range', 'end_date')],
    [Input('upload-data', 'contents')]
)
def update_table_and_filters(contents):
    if contents is None:
        return "", [], None, None
    
    # Parsear o CSV
    df = parse_contents(contents)
    
    # Criar tabela interativa para exibir os dados
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white'
        },
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white'
        }
    )
    
    # Gerar opções de variáveis para o dropdown
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    dropdown_options = [{'label': var, 'value': var} for var in numeric_columns]
    
    # Definir as datas mínima e máxima para o seletor de datas
    start_date = df['DATA'].min()
    end_date = df['DATA'].max()
    
    return table, dropdown_options, start_date, end_date

def definir_estacao(data):
    """Define a estação do ano com base na data."""
    mes = data.month
    dia = data.day
    
    if (mes == 3 and dia >= 21) or (3 < mes < 6) or (mes == 6 and dia < 21):
        return 'Outono'
    elif (mes == 6 and dia >= 21) or (6 < mes < 9) or (mes == 9 and dia < 23):
        return 'Inverno'
    elif (mes == 9 and dia >= 23) or (9 < mes < 12) or (mes == 12 and dia < 21):
        return 'Primavera'
    else:
        return 'Verão'
    
# Callback para definir o modo do app
@app.callback(
    [Output('exploratory-btn', 'active'),
     Output('time-series-btn', 'active')],
    [Input('exploratory-btn', 'n_clicks'),
     Input('time-series-btn', 'n_clicks')]
)
def set_app_mode(exploratory_clicks, time_series_clicks):
    global APP_MODE

    ctx = dash.callback_context
    triggered = ctx.triggered_id
    if triggered == "exploratory-btn":
        APP_MODE = "exploratory"
    elif triggered == "time-series-btn":
        APP_MODE = "time-series"

    return APP_MODE == "exploratory", APP_MODE == "time-series"

@app.callback(
    [Output('content-loading-text-container', 'style', allow_duplicate=True),
     Output('content-loading-text-container', 'children'),
     Output('content-container', 'style', allow_duplicate=True),
     Output('correlation-matrix', 'style', allow_duplicate=True),
     Output('time-series-plot-container', 'style', allow_duplicate=True),
     Output('variable-station-comparison', 'style', allow_duplicate=True),
     Output('scatter-plot', 'style', allow_duplicate=True),
     Output('histogram-plot-container', 'style', allow_duplicate=True)],
    [Input('upload-data', 'contents'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('variable-dropdown', 'value'),
     Input('exploratory-btn', 'n_clicks'),
     Input('time-series-btn', 'n_clicks'),
     Input('model-analysis-checkbox', 'value')],
     prevent_initial_call=True
)
def update_loading_text(contents, start_date, end_date, selected_variables, exploratory_clicks, time_series_clicks, model_analysis_checkbox):
    global APP_MODE

    if contents is None:
        return {'display': 'none'}, [], {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

    children = [
        html.P(
            "Carregando Análise Exploratória..." if APP_MODE == 'exploratory' else "Carregando Análise de Séries Temporais...",
            style={'text-align': 'center', 'padding': '150px' }
        )
    ]

    return {'display': 'block'}, children, {'display': 'none' }, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

@app.callback(
    [Output('correlation-matrix', 'figure'),
     Output('time-series-plot-container', 'children'),
     Output('variable-station-comparison', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('histogram-plot-container', 'children'),
     Output('correlation-matrix', 'style'),
     Output('time-series-plot-container', 'style'),
     Output('variable-station-comparison', 'style'),
     Output('scatter-plot', 'style'),
     Output('histogram-plot-container', 'style'),
     Output('content-loading-text-container', 'style'),
     Output('content-container', 'style')],
    [Input('upload-data', 'contents'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('variable-dropdown', 'value'),
     Input('exploratory-btn', 'n_clicks'),
     Input('time-series-btn', 'n_clicks'),
     Input('model-analysis-checkbox', 'value')]
)
def update_graphs(contents, start_date, end_date, selected_variables, exploratory_clicks, time_series_clicks, model_analysis_checkbox):
    global APP_MODE
    
    if contents is None:
        return {}, {}, {}, {}, [], {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

    # Verificar se selected_variables é None e se tornar uma lista vazia
    if selected_variables is None:
        selected_variables = []

    # Parsear o CSV
    df = parse_contents(contents)
    
    # Converter a coluna 'DATA' para datetime
    df['DATA'] = pd.to_datetime(df['DATA'])
    
    # Se start_date ou end_date forem None, use as datas mínimas e máximas do DataFrame
    if start_date is None:
        start_date = df['DATA'].min()
    if end_date is None:
        end_date = df['DATA'].max()
    
    # Filtrar os dados com base na seleção de datas
    df_filtered = df[(df['DATA'] >= start_date) & (df['DATA'] <= end_date)]
    
    # Substituir valores inválidos (-9999.0) por NaN
    df_filtered.replace(-9999.0, np.nan, inplace=True)
    
    # Inicializar valores de retorno
    fig_corr = {}
    fig_time_series = {}
    fig_station = {}
    fig_scatter = {}
    fig_histograms = []
    fig_time_series_by_station = {}
    style_corr = {'display': 'none'}
    style_time_series = {'display': 'none'}
    style_station = {'display': 'none'}
    style_scatter = {'display': 'none'}
    style_histogram = {'display': 'none'}
    time_series_plots = []

    # Análise exploratória
    if APP_MODE == 'exploratory':
        # Exibir a análise exploratória, ocultar séries temporais
        style_time_series = {'display': 'none'}
        style_station = {'display': 'none'}

        if len(selected_variables) == 0:
            # Matriz de correlação de todas as variáveis
            all_variables = df_filtered.select_dtypes(include=['float64', 'int64']).columns
            corr_matrix = df_filtered[all_variables].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect='auto', title="Matriz de Correlação")

            # Agrupando por mês para gerar o climograma de precipitação e temperatura
            df_filtered['MES'] = df_filtered['DATA'].dt.month
            climogram = df_filtered.groupby('MES').agg({
                'PRECIPITAÇÃO TOTAL': 'sum',  # Soma da precipitação
                'TEMPERATURA MEDIA': 'mean'   # Média da temperatura
            }).reset_index()

            # Criando o gráfico de barras para precipitação e sobrepondo a linha de temperatura
            fig_climogram = go.Figure()

            # Adiciona as barras de precipitação
            fig_climogram.add_trace(go.Bar(
                x=climogram['MES'],
                y=climogram['PRECIPITAÇÃO TOTAL'],
                name='Precipitação Total (mm)',
                marker_color='rgba(0, 123, 255, 0.7)',
                yaxis='y1'
            ))

            # Adiciona a linha para temperatura
            fig_climogram.add_trace(go.Scatter(
                x=climogram['MES'],
                y=climogram['TEMPERATURA MEDIA'],
                mode='lines+markers',
                name='Temperatura Média (°C)',
                line=dict(color='rgba(255, 100, 0, 0.8)', width=3),
                yaxis='y2'
            ))

            # Configurando o layout com dois eixos Y e rótulos mensais no eixo X
            fig_climogram.update_layout(
                title="Climograma: Precipitação e Temperatura Média por Mês",
                xaxis=dict(
                    title='Mês',
                    tickmode='array',
                    tickvals=list(range(1, 13)),
                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ),
                yaxis=dict(
                    title="Precipitação Total (mm)",
                    titlefont=dict(color='rgba(0, 123, 255, 0.7)'),
                    tickfont=dict(color='rgba(0, 123, 255, 0.7)'),
                    anchor='x',
                    side='left'
                ),
                yaxis2=dict(
                    title="Temperatura Média (°C)",
                    titlefont=dict(color='rgba(255, 100, 0, 0.8)'),
                    tickfont=dict(color='rgba(255, 100, 0, 0.8)'),
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.1, y=1.1),
                barmode='overlay'
            )

            # Definindo estilos para exibição dos gráficos
            style_corr = {'display': 'block'}
            style_station = {'display': 'block'}

            # Atualizando o valor do gráfico de precipitação e temperatura no retorno
            fig_station = fig_climogram

        if len(selected_variables) == 1:
            var = selected_variables[0]
            fig_boxplot = px.box(df_filtered, y=var, title=f"Boxplot de {var}")

            fig_histograms.append(dcc.Graph(figure=fig_boxplot))
            style_histogram = {'display': 'block'}
        
            if df_filtered['DATA'].dt.year.nunique() > 1:
                df_filtered['Estação'] = pd.cut(df_filtered['DATA'].dt.month % 12 + 1, bins=[0, 3, 6, 9, 12], labels=['Verão', 'Outono', 'Inverno', 'Primavera'])
                fig_station = px.bar(df_filtered.groupby('Estação')[selected_variables].mean().reset_index(), x='Estação', y=selected_variables, title="Comparação por Estação")
                style_station = {'display': 'block'}
        
        # Matriz de correlação (somente se mais de uma variável for selecionada)
        if len(selected_variables) > 1:
            df_selected = df_filtered[selected_variables]
            corr_matrix = df_selected.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect='auto', title="Matriz de Correlação")
            style_corr = {'display': 'block'}

        # Gerar um histograma para cada variável selecionada
        if len(selected_variables) >= 1:
            for var in selected_variables:
                fig_histogram = px.histogram(df_filtered, x=var, nbins=50, title=f"Histograma de {var}")
                # Adiciona cada histograma como um dcc.Graph dentro da lista
                fig_histograms.append(dcc.Graph(figure=fig_histogram))
            style_histogram = {'display': 'block'}

        # Gráfico de dispersão se houver duas variáveis
        if len(selected_variables) == 2:
            fig_scatter = px.scatter(df_filtered, x=selected_variables[0], y=selected_variables[1], title=f"Dispersão entre {selected_variables[0]} e {selected_variables[1]}")
            style_scatter = {'display': 'block'}


    elif APP_MODE == 'time-series':
        # Exibir séries temporais, ocultar análise exploratória
        style_corr = {'display': 'none'}
        style_scatter = {'display': 'none'}
        style_histogram = {'display': 'none'}

        variables = selected_variables.copy()

        # Caso default: Sem variáveis selecionadas, exibir série temporal da temperatura média
        if len(variables) == 0:
            variables.append('TEMPERATURA MEDIA')

        for idx, var in enumerate(variables):
            var_series = df_filtered[var].dropna()

            # Gráfico da série temporal original
            fig_time_series = px.line(df_filtered, x='DATA', y=var, title=f"Série Temporal de {var}")
            time_series_plots.append(
                dcc.Graph(
                    figure=fig_time_series,
                    style={'display': 'block'}
                )
            )
            
            # Criar uma cópia de df_filtered para a decomposição para evitar efeitos colaterais
            df_for_decomposition = df_filtered.copy()
            df_for_decomposition.set_index('DATA', inplace=True)  # Define 'DATA' como índice para a decomposição
            period_to_use = 8760

            if len(df_for_decomposition) < (period_to_use * 2):
                time_series_plots.append(
                    html.P(f"Número de Observações Insuficiente para Criar Gráfico de Tendência. Número Mínimo: {period_to_use * 2} | Número Atual: {len(df_for_decomposition)}.",
                            style={'text-align': 'center', 'padding': '75px' })
                )
            else:
                decomposition = seasonal_decompose(df_for_decomposition[var], model='additive', period=period_to_use)
                trend = decomposition.trend  # Extraindo a tendência

                # Gráfico da tendência
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=trend.index, 
                    y=trend, 
                    mode='lines', 
                    name='Tendência',
                    line=dict(color='orange')
                ))
                fig_trend.update_layout(
                    title=f"Tendência da Série Temporal de {var}",
                    xaxis_title="Data",
                    yaxis_title=f"{var}"
                )

                time_series_plots.append(
                    dcc.Graph(
                        figure=fig_trend,
                        style={'display': 'block'}
                    )
                )

            # Teste inicial de estacionaridade (sem diferenciação)
            adf_test = adfuller(var_series, regression='ct', autolag="AIC")
            kpss_test = kpss(var_series, regression='ct')

            adf_p_value = adf_test[1]
            kpss_p_value = kpss_test[1]

            estacionario = False
            diff_order = 0
            series_to_plot = None  # Inicialmente sem série para plotar
            test_results = html.Div()
            acf_pacf_plots = None

            # Verifica a estacionaridade na série original
            if adf_p_value < 0.05 and kpss_p_value > 0.05:
                estacionario = True
                # Exibir os resultados dos testes sem plotar a série
                test_results = html.Div([
                    html.H4("Resultados dos Testes de Estacionaridade", style={'textAlign': 'center'}),  # Título principal
                    html.H5("Série Original", style={'textAlign': 'center'}),
                    html.P(f"Estatística de Teste ADF: {adf_test[0]:.4f}", style={'textAlign': 'center'}),
                    html.P(f"p-valor ADF: {adf_p_value:.4f}", style={'textAlign': 'center'}),
                    html.P(f"Número de Lags ADF: {adf_test[2]}", style={'textAlign': 'center'}),
                    html.P(f"Estatística de Teste KPSS: {kpss_test[0]:.4f}", style={'textAlign': 'center'}),
                    html.P(f"p-valor KPSS: {kpss_p_value:.4f}", style={'textAlign': 'center'}),
                    html.P(f"Número de Lags KPSS: {kpss_test[2]}", style={'textAlign': 'center'})
                ])

            else:
                # Se não passou, aplica a primeira diferenciação e refaz os testes
                diff_order = 1
                diff1 = var_series.diff().dropna()
                adf_test_diff1 = adfuller(diff1, regression='c', autolag="AIC")
                kpss_test_diff1 = kpss(diff1, regression='c')

                if adf_test_diff1[1] < 0.05 and kpss_test_diff1[1] > 0.05:
                    estacionario = True
                    series_to_plot = diff1  # Série diferenciada de 1ª ordem
                    test_results = html.Div([
                        html.H4("Resultados dos Testes de Estacionaridade", style={'textAlign': 'center'}),  # Título principal
                        html.H5("Diferenciação de 1ª Ordem", style={'textAlign': 'center'}),
                        html.P(f"Estatística de Teste ADF: {adf_test_diff1[0]:.4f}", style={'textAlign': 'center'}),
                        html.P(f"p-valor ADF: {adf_test_diff1[1]:.4f}", style={'textAlign': 'center'}),
                        html.P(f"Número de Lags ADF: {adf_test_diff1[2]}", style={'textAlign': 'center'}),
                        html.P(f"Estatística de Teste KPSS: {kpss_test_diff1[0]:.4f}", style={'textAlign': 'center'}),
                        html.P(f"p-valor KPSS: {kpss_test_diff1[1]:.4f}", style={'textAlign': 'center'}),
                        html.P(f"Número de Lags KPSS: {kpss_test_diff1[2]}", style={'textAlign': 'center'})
                    ])

                    # Gerar os gráficos ACF e PACF para diff1
                    acf_values = acf(diff1, nlags=50)
                    pacf_values = pacf(diff1, nlags=50, method='ywmle')

                    acf_pacf_plots = go.Figure()
                    acf_pacf_plots.add_trace(go.Bar(x=np.arange(len(acf_values)), y=acf_values, name='ACF'))
                    acf_pacf_plots.add_trace(go.Bar(x=np.arange(len(pacf_values)), y=pacf_values, name='PACF', marker_color='orange'))

                    acf_pacf_plots.update_layout(
                        title="Funções de Autocorrelação (ACF) e Autocorrelação Parcial (PACF) - Diferença de 1ª Ordem",
                        xaxis_title="Lags",
                        yaxis_title="Valor",
                        barmode='group'
                    )

                else:
                    # Se ainda não passou, aplica a segunda diferenciação e refaz os testes
                    diff_order = 2
                    diff2 = diff1.diff().dropna()
                    adf_test_diff2 = adfuller(diff2, regression='c', autolag="AIC")
                    kpss_test_diff2 = kpss(diff2, regression='c')

                    if adf_test_diff2[1] < 0.05 and kpss_test_diff2[1] > 0.05:
                        estacionario = True
                        series_to_plot = diff2  # Série diferenciada de 2ª ordem
                        test_results = html.Div([
                            html.H4("Resultados dos Testes de Estacionaridade", style={'textAlign': 'center'}),  # Título principal
                            html.H5("Diferenciação de 2ª Ordem", style={'textAlign': 'center'}),
                            html.P(f"Estatística de Teste ADF: {adf_test_diff1[0]:.4f}", style={'textAlign': 'center'}),
                            html.P(f"p-valor ADF: {adf_test_diff1[1]:.4f}", style={'textAlign': 'center'}),
                            html.P(f"Número de Lags ADF: {adf_test_diff1[2]}", style={'textAlign': 'center'}),
                            html.P(f"Estatística de Teste KPSS: {kpss_test_diff1[0]:.4f}", style={'textAlign': 'center'}),
                            html.P(f"p-valor KPSS: {kpss_test_diff1[1]:.4f}", style={'textAlign': 'center'}),
                            html.P(f"Número de Lags KPSS: {kpss_test_diff1[2]}", style={'textAlign': 'center'})
                        ])

                        # Gerar os gráficos ACF e PACF para diff2
                        acf_values = acf(diff2, nlags=50)
                        pacf_values = pacf(diff2, nlags=50, method='ywmle')

                        acf_pacf_plots = go.Figure()
                        acf_pacf_plots.add_trace(go.Bar(x=np.arange(len(acf_values)), y=acf_values, name='ACF'))
                        acf_pacf_plots.add_trace(go.Bar(x=np.arange(len(pacf_values)), y=pacf_values, name='PACF', marker_color='orange'))

                        acf_pacf_plots.update_layout(
                            title="Funções de Autocorrelação (ACF) e Autocorrelação Parcial (PACF) - Diferença de 2ª Ordem",
                            xaxis_title="Lags",
                            yaxis_title="Valor",
                            barmode='group'
                        )

            # Plotar o gráfico de estacionaridade somente se a série passou após diferenciação
            if estacionario and series_to_plot is not None:
                estacionaridade_plot = go.Figure()

                estacionaridade_plot.add_trace(go.Scatter(
                    x=df_filtered["DATA"], 
                    y=series_to_plot, 
                    mode='lines', 
                    name=f"Diferença de {diff_order}ª Ordem"
                ))

                estacionaridade_plot.update_layout(
                    title=f"Estacionaridade: Diferença de {diff_order}ª Ordem para {var}",
                    xaxis_title="Data",
                    yaxis_title="Diferença" if diff_order > 0 else var,
                    legend_title="Série Diferenciada"
                )

                # Adiciona o gráfico de estacionaridade ao layout
                time_series_plots.append(
                    dcc.Graph(
                        figure=estacionaridade_plot,
                        style={'display': 'block'}
                    )
                )

            # Adiciona os resultados dos testes no layout
            time_series_plots.append(test_results)

            # Adiciona o gráfico de autocorrelação (ACF/PACF) ao layout se houver diferenciação
            if acf_pacf_plots:
                time_series_plots.append(
                    dcc.Graph(
                        figure=acf_pacf_plots,
                        style={'display': 'block', 'margin': 'auto'}
                    )
                )

            if df_filtered['DATA'].dt.year.nunique() > 1:
                # Extrair DIA e MES da coluna 'DATA'
                df_filtered['DIA'] = df_filtered['DATA'].dt.day
                df_filtered['MES'] = df_filtered['DATA'].dt.month

                # Criar um DataFrame de médias removendo colunas desnecessárias
                df_median = df_filtered.copy()
                df_median.drop(columns=['ESTACAO', 'HORA UTC', 'RADIACAO GLOBAL', 'DATA'], inplace=True)

                # Agrupar por DIA e MES para calcular a média das colunas numéricas
                df_median = df_median.groupby(['DIA', 'MES']).mean().reset_index()

                # Usar o ano das datas selecionadas no filtro
                selected_year = pd.to_datetime(start_date).year if start_date else 2024

                # Formatando as datas para o ano selecionado no filtro
                df_median['DATA_FORMATADA'] = pd.to_datetime(
                    {'day': df_median['DIA'], 'month': df_median['MES'], 'year': selected_year},
                    errors='coerce'  # Ignorar datas inválidas
                )

                # Filtrar valores inválidos (NaT)
                df_median = df_median.dropna(subset=['DATA_FORMATADA'])

                # Definir a estação do ano baseada nas datas
                df_median['ESTACAO'] = df_median['DATA_FORMATADA'].apply(definir_estacao)

                # Ordenar os dados por DATA_FORMATADA
                df_median = df_median.sort_values(by='DATA_FORMATADA', ascending=True)

                # Filtrar os dados com base nas datas selecionadas pelo filtro de data
                df_median_m = df_median[(df_median['DATA_FORMATADA'] >= pd.to_datetime(start_date)) & 
                                        (df_median['DATA_FORMATADA'] <= pd.to_datetime(end_date)) &
                                        ~( (df_median['DATA_FORMATADA'].dt.month == 12) & (df_median['DATA_FORMATADA'].dt.day >= 21) )]

                # Agora criar um gráfico para cada variável
                fig_time_series_by_station = go.Figure()

                # Adicionar uma série (linha) para cada estação dentro deste gráfico
                for estacao in df_median_m['ESTACAO'].unique():
                    dados_estacao = df_median_m[df_median_m['ESTACAO'] == estacao]

                    fig_time_series_by_station.add_trace(go.Scatter(
                        x=dados_estacao['DATA_FORMATADA'],
                        y=dados_estacao[var],
                        mode='lines',
                        name=estacao
                    ))

                # Ajustar o layout do gráfico para cada variável
                fig_time_series_by_station.update_layout(
                    title=f"Série Temporal de {var} por Estação",
                    xaxis_title="Mês",
                    yaxis_title=f"{var}",
                    legend_title="Estações",
                    showlegend=True,
                    xaxis=dict(
                        tickformat="%b" 
                    )
                )

                # Adicionar linhas verticais para transições de estação
                fig_time_series_by_station.add_vline(x=pd.Timestamp(f'{selected_year}-03-21'), line_dash="dash", line_color="gray")
                fig_time_series_by_station.add_vline(x=pd.Timestamp(f'{selected_year}-06-21'), line_dash="dash", line_color="gray")
                fig_time_series_by_station.add_vline(x=pd.Timestamp(f'{selected_year}-09-23'), line_dash="dash", line_color="gray")

                # Adicionar o gráfico de cada variável dentro da lista de gráficos
                time_series_plots.append(
                    dcc.Graph(
                        figure=fig_time_series_by_station,
                        style={'display': 'block'}
                    )
                )

            # Container Async para Análises de Modelagem
            model_analysis_enabled = 'enabled' in model_analysis_checkbox

            if model_analysis_enabled:
                # Placeholder para análises de modelagem
                time_series_plots.append(
                    html.Div(id={"type": "model-analysis-container", "variable": var, "order": diff_order}, children=[
                        html.P("Carregando Análises de Modelagem...", style={'text-align': 'center', 'padding': '75px' })
                    ])
                )

            if idx != len(variables) - 1:
                time_series_plots.append(html.Hr())


            style_time_series = {'display': 'block'}

    return fig_corr, time_series_plots, fig_station, fig_scatter, fig_histograms, style_corr, style_time_series, style_station, style_scatter, style_histogram, {'display': 'none'}, {'display': 'block'}

@app.callback(
    Output({'type': 'model-analysis-container', 'variable': MATCH, 'order': MATCH}, 'children'),
    [Input('upload-data', 'contents'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('variable-dropdown', 'value'),
     Input('exploratory-btn', 'n_clicks'),
     Input('time-series-btn', 'n_clicks'),
     Input('model-analysis-checkbox', 'value')],
    [State({'type': 'model-analysis-container', 'variable': MATCH, 'order': MATCH}, 'id')]
)
def update_model_analysis(contents, start_date, end_date, selected_variables, exploratory_clicks, time_series_clicks, model_analysis_checkbox, id):
    global APP_MODE

    children = []

    if contents is None:
        return children

    if APP_MODE == 'time-series':
        if 'enabled' in model_analysis_checkbox:
            var = id['variable']
            diff_order = id['order']

            model = MODEL_FOR_VARIABLE.get(var, None)

            if model:
                # Parsear o CSV
                df = parse_contents(contents)
                
                # Converter a coluna 'DATA' para datetime
                df['DATA'] = pd.to_datetime(df['DATA'])
                
                # Se start_date ou end_date forem None, use as datas mínimas e máximas do DataFrame
                if start_date is None:
                    start_date = df['DATA'].min()
                if end_date is None:
                    end_date = df['DATA'].max()
                
                # Filtrar os dados com base na seleção de datas
                df_filtered = df[(df['DATA'] >= start_date) & (df['DATA'] <= end_date)]
                
                # Substituir valores inválidos (-9999.0) por NaN
                df_filtered.replace(-9999.0, np.nan, inplace=True)

                data_to_fit = df_filtered[var].dropna()

                for _ in range(0, diff_order):
                    data_to_fit = data_to_fit.diff().dropna()
                
                # Estimando o modelo
                result = model.fit(data_to_fit)

                # Extrair o resumo como texto
                summary_text = result.summary().as_text()

                # Exibir o resumo do modelo ARIMA no layout
                arima_summary = html.Div([
                    html.H4(f"Resumo do Modelo ARIMA para {var}", style={'textAlign': 'center'}),
                    dcc.Markdown(
                        f"```\n{summary_text}\n```",
                        style={
                            'whiteSpace': 'pre-wrap', 
                            'fontFamily': 'monospace',
                            'textAlign': 'center',
                            'margin': '0 auto',
                            'display': 'inline-block',
                        }
                    )
                ], style={'textAlign': 'center'})

                # Adicionar o resumo à lista de elementos a serem exibidos
                children.append(arima_summary)

                # Diagnóstico 1. Resíduos
                res = result.resid()

                residuos_plot = go.Figure()
                residuos_plot.add_trace(go.Scatter(
                    x=df_filtered['DATA'], 
                    y=res, 
                    mode='lines', 
                    name="Resíduos", 
                    line=dict(color='blue')
                ))
                residuos_plot.add_hline(y=0, line_dash="dash", line_color="red", name="Linha Zero")
                residuos_plot.update_layout(
                    title="Resíduos do Modelo ARIMA",
                    xaxis_title="Tempo",
                    yaxis_title="Resíduos"
                )
                children.append(
                    dcc.Graph(figure=residuos_plot)
                )

                # Diagnóstico 2. Autocorrelações dos resíduos
                acf_res = acf(res, nlags=15)
                pacf_res = pacf(res, nlags=15)

                acf_pacf_res_plot = go.Figure()
                acf_pacf_res_plot.add_trace(go.Bar(x=np.arange(len(acf_res)), y=acf_res, name="ACF dos Resíduos"))
                acf_pacf_res_plot.add_trace(go.Bar(x=np.arange(len(pacf_res)), y=pacf_res, name="PACF dos Resíduos", marker_color='orange'))
                acf_pacf_res_plot.update_layout(
                    title="ACF e PACF dos Resíduos",
                    xaxis_title="Lags",
                    yaxis_title="Correlação",
                    barmode='group'
                )
                children.append(
                    dcc.Graph(figure=acf_pacf_res_plot)
                )

                # Diagnóstico 3. QQPlot dos resíduos
                qq_data = stats.probplot(res, dist="norm")
                theoretical_quantiles = qq_data[0][0]
                ordered_values = qq_data[0][1]

                qq_plot = go.Figure()

                # Adicionando pontos do QQPlot
                qq_plot.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=ordered_values,
                    mode='markers',
                    name='Quantis',
                    marker=dict(color='blue', size=6)
                ))

                # Adicionando a linha de referência (linha ideal)
                qq_plot.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=theoretical_quantiles,
                    mode='lines',
                    name='Linha de Referência',
                    line=dict(color='red', dash='dash')
                ))

                # Configurando o layout do gráfico
                qq_plot.update_layout(
                    title="Q-Q Plot dos Resíduos",
                    xaxis_title="Quantis Teóricos",
                    yaxis_title="Valores Ordenados",
                    showlegend=True
                )

                # Adicionar o gráfico ao layout do Dash
                children.append(
                    dcc.Graph(figure=qq_plot)
                )

                # Previsão para os próximos 6 meses
                forecast = result.predict(n_periods=6 * 30 * 24)

                last_value = df_filtered[var].iloc[-1]
                forecast_original = forecast.cumsum() + last_value

                future_dates = pd.date_range(start=df_filtered['DATA'].iloc[-1], periods=len(forecast_original) + 1, freq='h')[1:]

                # Determinar o período de previsão para o título
                start_forecast_date = future_dates[0].strftime('%d/%m/%Y')
                end_forecast_date = future_dates[-1].strftime('%d/%m/%Y')

                forecast_plot = go.Figure()
                forecast_plot.add_trace(go.Scatter(
                    x=df_filtered['DATA'], 
                    y=df_filtered[var], 
                    mode='lines', 
                    name="Original"
                ))
                forecast_plot.add_trace(go.Scatter(
                    x=future_dates, 
                    y=forecast_original, 
                    mode='lines', 
                    name="Previsão", 
                    line=dict(color='orange')
                ))
                forecast_plot.update_layout(
                    title=f"Previsão para o Período: {start_forecast_date} - {end_forecast_date}",
                    xaxis_title="Tempo",
                    yaxis_title=var
                )

                # Adicionar o gráfico ao layout do Dash
                children.append(
                    dcc.Graph(figure=forecast_plot)
                )
            else:
                return [
                    html.P(f"Modelo Não Encontrado Para Variável: {var}", style={'text-align': 'center', 'padding': '75px' })
                ]


    return children


if __name__ == '__main__':
    app.run_server(debug=True)
