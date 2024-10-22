import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import io
import base64
import numpy as np
from dash import dash_table
import plotly.graph_objects as go

# Inicializando o Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    html.Div(id='table-container', style={'margin': '20px'}),
    
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
    
    # Exibição dos gráficos
    dcc.Graph(id='correlation-matrix', style={'display': 'none'}),
    html.Div(id='time-series-plot-container', style={'display': 'none'}),
    dcc.Graph(id='time-series-graph', style={'display': 'none'}),
    dcc.Graph(id='time-series-by-station-graph', style={'display': 'none'}),
    html.Div(id='time-series-by-station-container', style={'display': 'none'}),

    # Gráfico de comparação por estação
    html.Div([
        dcc.Graph(id='variable-station-comparison', style={'display': 'none'}),
        html.P("", 
               style={'text-align': 'center', 'font-size': '12px', 'color': '#7f8c8d', 'margin-top': '10px'})
    ]),

    # Gráficos extras para análise exploratória
    dcc.Graph(id='scatter-plot', style={'display': 'none'}),
    html.Div(id='histogram-plot-container', style={'display': 'none'})
])

# Função para parsear o CSV
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

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
     Output('time-series-by-station-container', 'children'),
     Output('time-series-by-station-container', 'style'),
     Output('exploratory-btn', 'active'),
     Output('time-series-btn', 'active')],
    [Input('upload-data', 'contents'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('variable-dropdown', 'value'),
     Input('exploratory-btn', 'n_clicks'),
     Input('time-series-btn', 'n_clicks')]
)
def update_graphs(contents, start_date, end_date, selected_variables, exploratory_clicks, time_series_clicks):
    if contents is None:
        return {}, {}, {}, {}, [], {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {}, {'display': 'none'}, True, False

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
    style_time_series_by_station = {'display': 'none'}
    time_series_by_station_graphs = []
    time_series_plots = []

    # Verificar qual botão foi clicado para determinar o modo de análise
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Análise exploratória
    if button_id == 'exploratory-btn' or exploratory_clicks >= time_series_clicks:
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


    elif button_id == 'time-series-btn' or time_series_clicks > exploratory_clicks:
        # Exibir séries temporais, ocultar análise exploratória
        style_corr = {'display': 'none'}
        style_scatter = {'display': 'none'}
        style_histogram = {'display': 'none'}

        # Caso default: Sem variáveis selecionadas, exibir série temporal da temperatura média
        if len(selected_variables) == 0:
            fig_time_series = px.line(df_filtered, x='DATA', y='TEMPERATURA MEDIA', title="Série Temporal da Temperatura Média")
            time_series_plots.append(
                dcc.Graph(
                    figure=fig_time_series,
                    style={'display': 'block'}
                )
            )
            style_time_series = {'display': 'block'}

        # Se uma ou mais variáveis forem selecionadas
        elif len(selected_variables) >= 1:
            for var in selected_variables:
                fig_time_series = px.line(df_filtered, x='DATA', y=var, title=f"Série Temporal de {var}")
                time_series_plots.append(
                    dcc.Graph(
                        figure=fig_time_series,
                        style={'display': 'block'}
                    )
                )
            style_time_series = {'display': 'block'}

            if df_filtered['DATA'].dt.year.nunique() > 1:
                for var in selected_variables:
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
                    time_series_by_station_graphs.append(
                        dcc.Graph(
                            figure=fig_time_series_by_station,
                            style={'display': 'block'}
                        )
                    )

                style_time_series_by_station = {'display': 'block'}

    return fig_corr, time_series_plots, fig_station, fig_scatter, fig_histograms, style_corr, style_time_series, style_station, style_scatter, style_histogram, time_series_by_station_graphs, style_time_series_by_station, button_id == 'exploratory-btn', button_id == 'time-series-btn'


if __name__ == '__main__':
    app.run_server(debug=True)
