import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##########1.1##########

# Считывание данных из CSV файлов
visitation_df = pd.read_csv('visitation.csv')
orders_df = pd.read_csv('orders.csv')
expenses_df = pd.read_csv('expenses.csv')

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

# Сохранение обработанных данных в CSV файлы
visitation_df.to_csv('processed_visitation.csv', index=False)
orders_df.to_csv('processed_orders.csv', index=False)
expenses_df.to_csv('processed_expenses.csv', index=False)

print("Данные успешно обработаны и сохранены в CSV файлы.")

##########1.3##########

# Функция для создания набора данных с профилями пользователей
def create_user_profiles():
    # Считывание данных из CSV файлов
    sessions_df = pd.read_csv('sessions.csv')
    advertising_costs_df = pd.read_csv('advertising_costs.csv')

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

    # Сохранение результата в CSV файл
    user_profiles_df.to_csv('user_profiles.csv', index=False)

    print("Набор данных с профилями пользователей успешно создан и сохранен в CSV файл.")

# Создание набора данных с профилями пользователей
create_user_profiles() 
