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

# Загрузка данных для дашборда
visitation_df = pd.read_sql_table('visitation', con=engine)
orders_df = pd.read_sql_table('orders', con=engine)
expenses_df = pd.read_sql_table('expenses', con=engine)

# Инициализация Dash приложения
app = dash.Dash(__name__)

# Описание макета дашборда
app.layout = html.Div([
    html.H1("Аналитический Дашборд"),
    
    # Добавление элементов управления для выбора метрики
    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'Количество посещений', 'value': 'visits'},
            {'label': 'Количество заказов', 'value': 'orders'},
            {'label': 'Расходы', 'value': 'expenses'}
        ],
        value='visits'  # Значение по умолчанию
    ),
    
    # Отображение графика выбранной метрики
    dcc.Graph(id='metric-graph')
])

# Описание обновления графика при изменении выбранной метрики
@app.callback(
    Output('metric-graph', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_graph(metric):
    if metric == 'visits':
        data = visitation_df.groupby('date')['user_id'].count().reset_index()
        title = 'Количество посещений по датам'
    elif metric == 'orders':
        data = orders_df.groupby('date')['order_id'].count().reset_index()
        title = 'Количество заказов по датам'
    elif metric == 'expenses':
        data = expenses_df.groupby('date')['cost'].sum().reset_index()
        title = 'Расходы по датам'
    
    figure = {
        'data': [
            {'x': data['date'], 'y': data[metric], 'type': 'line', 'name': metric}
        ],
        'layout': {
            'title': title,
            'xaxis': {'title': 'Дата'},
            'yaxis': {'title': metric}
        }
    }
    
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
