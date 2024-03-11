import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sqlalchemy import create_engine

# Параметры подключения к базе данных
DB_USER = 'your_username'
DB_PASSWORD = 'your_password'
DB_HOST = 'localhost'
DB_PORT = '5432'  # Порт базы данных
DB_NAME = 'Users'

# Создание строки подключения
connection_str = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Подключение к базе данных
engine = create_engine(connection_str)

# Загрузка данных для маркетингового анализа
sessions_df = pd.read_sql_table('sessions', con=engine)
advertising_costs_df = pd.read_sql_table('advertising_costs', con=engine)

# Расчет общей суммы расходов на рекламу
total_ad_costs = advertising_costs_df['costs'].sum()

# Расчет суммы расходов на рекламу по каждому источнику
ad_costs_by_channel = advertising_costs_df.groupby('channel')['costs'].sum()

# Расчет суммы расходов на рекламу по времени
ad_costs_by_date = advertising_costs_df.groupby('date')['costs'].sum()

# Расчет средней стоимости привлечения одного покупателя из каждого источника
avg_acquisition_cost_by_channel = ad_costs_by_channel / sessions_df[sessions_df['is_paying'] == True].groupby('channel')['user_id'].nunique()

# Расчет ROI по времени
revenue_df = pd.read_sql_table('revenue', con=engine)  # Загрузка данных о доходах
roi_by_date = ((revenue_df.groupby('date')['revenue'].sum() - ad_costs_by_date) / ad_costs_by_date) * 100

# Расчет ROI с разбивкой по устройствам
roi_by_device = []
for device in sessions_df['device'].unique():
    revenue_by_device = revenue_df.merge(sessions_df[sessions_df['device'] == device], on='user_id')['revenue'].sum()
    ad_costs_by_device = advertising_costs_df[advertising_costs_df['device'] == device]['costs'].sum()
    roi_by_device.append(((revenue_by_device - ad_costs_by_device) / ad_costs_by_device) * 100)

# Расчет ROI с разбивкой по странам
roi_by_country = []
for country in sessions_df['country'].unique():
    revenue_by_country = revenue_df.merge(sessions_df[sessions_df['country'] == country], on='user_id')['revenue'].sum()
    ad_costs_by_country = advertising_costs_df[advertising_costs_df['country'] == country]['costs'].sum()
    roi_by_country.append(((revenue_by_country - ad_costs_by_country) / ad_costs_by_country) * 100)

# Расчет ROI с разбивкой по рекламным каналам
roi_by_channel = []
for channel in sessions_df['channel'].unique():
    revenue_by_channel = revenue_df.merge(sessions_df[sessions_df['channel'] == channel], on='user_id')['revenue'].sum()
    ad_costs_by_channel = advertising_costs_df[advertising_costs_df['channel'] == channel]['costs'].sum()
    roi_by_channel.append(((revenue_by_channel - ad_costs_by_channel) / ad_costs_by_channel) * 100)

# Создание интерфейса для отображения маркетингового анализа
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Маркетинговый анализ"),
    
    html.Div([
        html.Div([
            html.H3("Общие расходы на рекламу"),
            html.P(f"{total_ad_costs} у.е.")
        ], className="six columns"),
        
        html.Div([
            html.H3("Расходы на рекламу по источникам"),
            dcc.Graph(id='ad-costs-by-channel-graph', figure={'data': [{'x': ad_costs_by_channel.index, 'y': ad_costs_by_channel.values, 'type': 'bar'}]})
        ], className="six columns")
    ], className="row"),
    
    html.Div([
        html.Div([
            html.H3("Средняя стоимость привлечения одного покупателя"),
            dcc.Graph(id='avg-acquisition-cost-by-channel-graph', figure={'data': [{'x': avg_acquisition_cost_by_channel.index, 'y': avg_acquisition_cost_by_channel.values, 'type': 'bar'}]})
        ], className="six columns"),
        
        html.Div([
            html.H3("ROI по времени"),
            dcc.Graph(id='roi-by-date-graph', figure={'data': [{'x': roi_by_date.index, 'y': roi_by_date.values, 'type': 'line'}]})
        ], className="six columns")
    ], className="row"),
    
    html.Div([
        html.Div([
            html.H3("ROI с разбивкой по устройствам"),
            dcc.Graph(id='roi-by-device-graph', figure={'data': [{'x': sessions_df['device'].unique(), 'y': roi_by_device, 'type': 'bar'}]})
        ], className="six columns"),
        
        html.Div([
            html.H3("ROI с разбивкой по странам"),
            dcc.Graph(id='roi-by-country-graph', figure={'data': [{'x': sessions_df['country'].unique(), 'y': roi_by_country, 'type': 'bar'}]})
        ], className="six columns")
    ], className="row"),
    
    html.Div([
        html.Div([
            html.H3("ROI с разбивкой по рекламным каналам"),
            dcc.Graph(id='roi-by-channel-graph', figure={'data': [{'x': sessions_df['channel'].unique(), 'y': roi_by_channel, 'type': 'bar'}]})
        ], className="six columns")
    ], className="row")
])

if __name__ == '__main__':
    app.run_server(debug=True)
