import dash
import dash_core_components as dcc
import dash_html_components as html
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

# Загрузка данных
visitation_df = pd.read_sql('SELECT * FROM visitation', engine)
orders_df = pd.read_sql('SELECT * FROM orders', engine)
expenses_df = pd.read_sql('SELECT * FROM expenses', engine)

# Создание экземпляра приложения Dash
app = dash.Dash(__name__)

# Описание макета дашборда
app.layout = html.Div([
    html.H1('Отчет о проделанной работе'),

    html.Div([
        html.H2('График посещаемости'),
        dcc.Graph(id='visitation-graph'),
    ]),

    html.Div([
        html.H2('График заказов'),
        dcc.Graph(id='orders-graph'),
    ]),

    html.Div([
        html.H2('График расходов'),
        dcc.Graph(id='expenses-graph'),
    ]),
])

# Callback-функции для обновления графиков при изменении данных
@app.callback(
    Output('visitation-graph', 'figure'),
    Output('orders-graph', 'figure'),
    Output('expenses-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graphs(n):
    # Обновление графика посещаемости
    visitation_fig = px.line(visitation_df, x='date', y='visits', title='Посещаемость')

    # Обновление графика заказов
    orders_fig = px.bar(orders_df, x='date', y='orders', title='Заказы')

    # Обновление графика расходов
    expenses_fig = px.area(expenses_df, x='date', y='expenses', title='Расходы')

    return visitation_fig, orders_fig, expenses_fig

# Запуск сервера Dash
if __name__ == '__main__':
    app.run_server(debug=True)

    
##############################################  ИЛИ  ##############################################


import os
import shutil

# Создание директории для сохранения результатов
if not os.path.exists('Results'):
    os.makedirs('Results')

# Сохранение результатов предварительной обработки данных в файлы .ipynb и .py
visitation_df.to_pickle('Results/visitation_processed.pkl')
orders_df.to_pickle('Results/orders_processed.pkl')
expenses_df.to_pickle('Results/expenses_processed.pkl')

# Создание файла .ipynb для отчета о предварительной обработке данных
with open('Results/Preprocessing_Report.ipynb', 'w') as f:
    f.write("# Preprocessing Report\n\n")
    f.write("This notebook contains the preprocessing report for the data.\n\n")
    f.write("## Visitations\n")
    f.write(visitation_df.head().to_markdown())
    f.write("\n\n## Orders\n")
    f.write(orders_df.head().to_markdown())
    f.write("\n\n## Expenses\n")
    f.write(expenses_df.head().to_markdown())

# Создание архива Data.zip
shutil.make_archive('Data', 'zip', 'Results')

# Создание файла Readme.txt с описанием содержимого архива
with open('Readme.txt', 'w') as f:
    f.write("Содержимое архива Data.zip:\n")
    f.write("- Results/: Папка с результатами работы\n")
    f.write("- Preprocessing_Report.ipynb: Отчет о предварительной обработке данных\n")
    f.write("- visitation_processed.pkl: Обработанные данные о посещениях\n")
    f.write("- orders_processed.pkl: Обработанные данные о заказах\n")
    f.write("- expenses_processed.pkl: Обработанные данные о расходах\n")
