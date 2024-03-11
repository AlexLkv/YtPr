import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt


##########1.1##########


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

# Считывание данных из таблицы visitation
visitation_df = pd.read_sql_table('visitation', con=engine)

# Считывание данных из таблицы orders
orders_df = pd.read_sql_table('orders', con=engine)

# Считывание данных из таблицы expenses
expenses_df = pd.read_sql_table('expenses', con=engine)


##########1.2##########


# Предварительная обработка данных
def preprocess_data(df):
    # Обработка пропусков
    df.fillna(0, inplace=True)  # Заполнение пропусков нулями

    # Обработка дубликатов
    df.drop_duplicates(inplace=True)

    # Обработка аномальных значений
    for column in df.columns:
        if df[column].dtype != 'object':  # Проверка только числовых столбцов
            # Визуализация данных для обнаружения выбросов
            sns.boxplot(x=df[column])
            plt.title(column)
            plt.show()

            # Удаление выбросов
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            threshold = 1.5 * iqr
            df = df.loc[(df[column] > (q1 - threshold)) & (df[column] < (q3 + threshold))]

    # Приведение данных к приемлемому формату
    for column in df.columns:
        if df[column].dtype == 'object':  # Проверка на столбцы с типом object
            df[column] = pd.to_datetime(df[column], errors='coerce')

    return df


# Применение предварительной обработки к каждому DataFrame
visitation_df = preprocess_data(visitation_df)
orders_df = preprocess_data(orders_df)
expenses_df = preprocess_data(expenses_df)

# Обновление данных в базе данных
visitation_df.to_sql('visitation', engine, if_exists='replace', index=False)
orders_df.to_sql('orders', engine, if_exists='replace', index=False)
expenses_df.to_sql('expenses', engine, if_exists='replace', index=False)

print("Данные успешно обновлены в базе данных.")


##########1.3##########


# Параметры подключения к базе данных
DB_USER = 'your_username'
DB_PASSWORD = 'your_password'
DB_HOST = 'localhost'
DB_PORT = '5432'  # Порт базы данных
DB_NAME = 'Users'

# Создание строки подключения
connection_str = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Функция для формирования набора данных с профилями пользователей
def create_user_profiles():
    # Подключение к базе данных
    engine = create_engine(connection_str)

    # Загрузка данных о сессиях пользователей
    sessions_df = pd.read_sql_table('sessions', con=engine)

    # Загрузка данных о расходах на рекламу
    advertising_costs_df = pd.read_sql_table('advertising_costs', con=engine)

    # Определение даты первого посещения каждого пользователя
    first_session_dates = sessions_df.groupby('user_id')['timestamp'].min().reset_index()
    first_session_dates.rename(columns={'timestamp': 'first_session_timestamp'}, inplace=True)

    # Рассчет средней стоимости привлечения пользователей в день
    average_user_acquisition_cost_per_day = advertising_costs_df['cost'].sum() / len(
        advertising_costs_df['date'].unique())

    # Объединение данных о первой сессии с данными о пользователях
    user_profiles_df = first_session_dates.merge(sessions_df, on='user_id')

    # Добавление полей: устройство, регион, рекламный источник
    user_profiles_df = user_profiles_df.merge(advertising_costs_df, on='campaign_id')
    user_profiles_df.drop_duplicates(inplace=True)  # Удаление возможных дубликатов

    # Сохранение результата в базе данных
    user_profiles_df.to_sql('user_profiles', engine, if_exists='replace', index=False)

    print("Набор данных с профилями пользователей успешно создан.")

# Функция для расчета ROI
def calculate_roi(revenue_df, costs_df):
    # Группировка доходов по каналам рекламы и вычисление общей прибыли
    revenue_by_channel = revenue_df.groupby('channel')['revenue'].sum()
    
    # Группировка расходов по каналам рекламы и вычисление общих расходов
    costs_by_channel = costs_df.groupby('channel')['costs'].sum()
    
    # Расчет ROI для каждого канала
    roi_by_channel = ((revenue_by_channel - costs_by_channel) / costs_by_channel) * 100
    
    return roi_by_channel

# Функция для расчета LTV
def calculate_ltv(revenue_df, acquisition_cost_df):
    # Группировка доходов по пользователям и вычисление общей прибыли
    revenue_by_user = revenue_df.groupby('user_id')['revenue'].sum()
    
    # Расчет средней стоимости привлечения пользователя
    average_acquisition_cost = acquisition_cost_df['cost'].mean()
    
    # Расчет LTV для каждого пользователя
    ltv_by_user = revenue_by_user - average_acquisition_cost
    
    return ltv_by_user

# Функция для расчета удержания и конверсии
def calculate_retention_and_conversion(users_df, sessions_df):
    # Определение уникальных пользователей за каждый день
    daily_users = sessions_df.groupby(pd.Grouper(key='timestamp', freq='D'))['user_id'].nunique()
    
    # Расчет удержания пользователей
    retention_rate = daily_users / users_df.shape[0] * 100
    
    # Расчет конверсии
    conversion_rate = users_df.shape[0] / daily_users
    
    return retention_rate, conversion_rate

# Создание набора данных с профилями пользователей
create_user_profiles()
