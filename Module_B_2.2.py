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

# Загрузка данных для исследовательского анализа
visitation_df = pd.read_sql_table('visitation', con=engine)

# Вычисление DAU (Daily Active Users)
dau = visitation_df.groupby('date')['user_id'].nunique()

# Вычисление WAU (Weekly Active Users)
wau = visitation_df.groupby(visitation_df['date'].dt.strftime('%U'))['user_id'].nunique()

# Вычисление MAU (Monthly Active Users)
mau = visitation_df.groupby(visitation_df['date'].dt.strftime('%Y-%m'))['user_id'].nunique()

# Вычисление среднего времени, проведенного в приложении
visitation_df['duration'] = visitation_df['end_time'] - visitation_df['start_time']
average_duration = visitation_df['duration'].mean()

# Создание интерфейса для отображения метрик
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Исследовательский анализ данных"),
    
    html.Div([
        html.Div([
            html.H3("DAU (Daily Active Users)"),
            dcc.Graph(id='dau-graph', figure={'data': [{'x': dau.index, 'y': dau.values, 'type': 'line'}]})
        ], className="six columns"),
        
        html.Div([
            html.H3("WAU (Weekly Active Users)"),
            dcc.Graph(id='wau-graph', figure={'data': [{'x': wau.index, 'y': wau.values, 'type': 'line'}]})
        ], className="six columns"),
    ], className="row"),
    
    html.Div([
        html.Div([
            html.H3("MAU (Monthly Active Users)"),
            dcc.Graph(id='mau-graph', figure={'data': [{'x': mau.index, 'y': mau.values, 'type': 'line'}]})
        ], className="six columns"),
        
        html.Div([
            html.H3("Среднее время в приложении"),
            html.P(f"{average_duration} секунд")
        ], className="six columns")
    ], className="row")
])

if __name__ == '__main__':
    app.run_server(debug=True)
