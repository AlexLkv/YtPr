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

# Функция для формирования набора данных с профилями пользователей
def create_user_profiles():
    # Загрузка данных о сессиях пользователей
    sessions_df = pd.read_sql_table('sessions', con=engine)

    # Загрузка данных о расходах на рекламу
    advertising_costs_df = pd.read_sql_table('advertising_costs', con=engine)

    # Определение стран, из которых приходят посетители
    countries = sessions_df['country'].unique()

    # Расчет количества платящих пользователей по странам
    paying_users_by_country = sessions_df[sessions_df['is_paying'] == True].groupby('country')['user_id'].nunique()

    # Определение устройств, которыми пользуются пользователи
    devices = sessions_df['device'].unique()

    # Расчет количества платящих пользователей по устройствам
    paying_users_by_device = sessions_df[sessions_df['is_paying'] == True].groupby('device')['user_id'].nunique()

    # Определение рекламных каналов привлечения пользователей
    advertising_channels = advertising_costs_df['channel'].unique()

    # Расчет количества платящих пользователей по рекламным каналам
    paying_users_by_channel = sessions_df[sessions_df['is_paying'] == True].groupby('channel')['user_id'].nunique()

    # Сохранение результатов в базе данных
    user_profiles_df = pd.DataFrame({
        'country': countries,
        'paying_users': paying_users_by_country
    }).reset_index()
    user_profiles_df.to_sql('user_profiles', engine, if_exists='replace', index=False)

    return countries, paying_users_by_country, devices, paying_users_by_device, advertising_channels, paying_users_by_channel

# Вызов функции для формирования профилей пользователей
countries, paying_users_by_country, devices, paying_users_by_device, advertising_channels, paying_users_by_channel = create_user_profiles()

# Создание интерфейса для отображения профилей пользователей
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Профили пользователей"),
    
    html.Div([
        html.Div([
            html.H3("Страны посетителей"),
            dcc.Graph(id='countries-graph', figure={'data': [{'x': countries, 'y': paying_users_by_country, 'type': 'bar'}]})
        ], className="six columns"),
        
        html.Div([
            html.H3("Устройства пользователей"),
            dcc.Graph(id='devices-graph', figure={'data': [{'x': devices, 'y': paying_users_by_device, 'type': 'bar'}]})
        ], className="six columns")
    ], className="row"),
    
    html.Div([
        html.Div([
            html.H3("Рекламные каналы привлечения"),
            dcc.Graph(id='channels-graph', figure={'data': [{'x': advertising_channels, 'y': paying_users_by_channel, 'type': 'bar'}]})
        ], className="twelve columns")
    ], className="row")
])

if __name__ == '__main__':
    app.run_server(debug=True)
