import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pickle

# Загрузка данных
orders_df = pd.read_csv('orders.csv')
visitation_df = pd.read_csv('visitation.csv')

# Преобразование данных к необходимым типам
orders_df['Event Dt'] = pd.to_datetime(orders_df['Event Dt'])
visitation_df['Session Start'] = pd.to_datetime(visitation_df['Session Start'])
visitation_df['Session End'] = pd.to_datetime(visitation_df['Session End'])

# Объединение данных по User Id для анализа RFM
grouped_orders = orders_df.groupby('User Id').agg({'Event Dt': 'max', 'Revenue': ['count', 'sum']}).reset_index()
grouped_orders.columns = ['User Id', 'Last Purchase Date', 'Frequency', 'Monetary']

# Рассчет Recency (R)
current_date = orders_df['Event Dt'].max()
grouped_orders['Recency'] = (current_date - grouped_orders['Last Purchase Date']).dt.days

# Проверим результат
print(grouped_orders.head())
# Определение квантилей для каждой метрики
quantiles = grouped_orders[['Recency', 'Frequency', 'Monetary']].quantile(q=[0.25, 0.5, 0.75]).to_dict()


# Функция для сегментации пользователей на основе RFM метрик
def rfm_segmentation(row):
    r_score = 4 if row['Recency'] <= quantiles['Recency'][0.25] else \
        3 if row['Recency'] <= quantiles['Recency'][0.5] else \
            2 if row['Recency'] <= quantiles['Recency'][0.75] else 1

    f_score = 4 if row['Frequency'] >= quantiles['Frequency'][0.75] else \
        3 if row['Frequency'] >= quantiles['Frequency'][0.5] else \
            2 if row['Frequency'] >= quantiles['Frequency'][0.25] else 1

    m_score = 4 if row['Monetary'] >= quantiles['Monetary'][0.75] else \
        3 if row['Monetary'] >= quantiles['Monetary'][0.5] else \
            2 if row['Monetary'] >= quantiles['Monetary'][0.25] else 1

    return f"{r_score}{f_score}{m_score}"


# Применение функции для создания нового столбца с сегментами
grouped_orders['RFM Segment'] = grouped_orders.apply(rfm_segmentation, axis=1)

# Проверим результат
print(grouped_orders.head())
from scipy.stats import f_oneway

# Группировка пользователей по сегментам
segment_groups = grouped_orders.groupby('RFM Segment')['Frequency'].count()

# Проверка гипотезы различия конверсии между сегментами
segments = grouped_orders['RFM Segment'].unique()
anova_results = {}

for segment in segments:
    segment_data = grouped_orders[grouped_orders['RFM Segment'] == segment]['Frequency']
    anova_results[segment] = segment_data

# Выполнение ANOVA теста
f_statistic, p_value = f_oneway(*anova_results.values())

# Вывод результатов
print("F-statistic:", f_statistic)
print("P-value:", p_value)

# Интерпретация результатов
alpha = 0.05
if p_value < alpha:
    print("Отвергаем нулевую гипотезу: существует статистически значимая разница в конверсии между сегментами.")
else:
    print(
        "Нет достаточных доказательств для отвержения нулевой гипотезы: разница в конверсии между сегментами статистически незначима.")

######################3.2
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Подготовка данных
X = grouped_orders[['Recency', 'Frequency', 'Monetary']]
y = grouped_orders['Monetary']  # Предполагаем, что выручка является целевой переменной

# Разделение выборки на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение моделей
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)

decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(X_train, y_train)

# Прогнозирование на валидационной выборке
linear_reg_predictions = linear_reg_model.predict(X_test)
random_forest_predictions = random_forest_model.predict(X_test)
decision_tree_predictions = decision_tree_model.predict(X_test)

# Оценка моделей
linear_reg_rmse = mean_squared_error(y_test, linear_reg_predictions, squared=False)
random_forest_rmse = mean_squared_error(y_test, random_forest_predictions, squared=False)
decision_tree_rmse = mean_squared_error(y_test, decision_tree_predictions, squared=False)

print("RMSE для линейной регрессии:", linear_reg_rmse)
print("RMSE для случайного леса:", random_forest_rmse)
print("RMSE для дерева решений:", decision_tree_rmse)
#
# RMSE для линейной регрессии: 2.7112184262483235e-14
# RMSE для случайного леса: 0.1421362534972855
# RMSE для дерева решений: 0.38830954067725293
# Исходя из этих результатов, наиболее оптимальной моделью для прогнозирования выручки от пользователей является модель линейной регрессии


###################3.3


# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
#
# # Создание пайплайна для преобразования данных и обучения модели
# pipeline = Pipeline([
#     ('poly_features', PolynomialFeatures()),
#     ('scaler', StandardScaler()),
#     ('linear_reg', LinearRegression())
# ])
#
# # Определение сетки параметров для подбора гиперпараметров
# param_grid = {
#     'poly_features__degree': [2, 3],  # Попробуем полиномиальные признаки второй и третьей степени
# }
#
# # Подбор гиперпараметров с использованием кросс-валидации
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)
#
# # Получение лучшей модели
# best_model = grid_search.best_estimator_
#
# # Прогнозирование на тестовой выборке
# predictions = best_model.predict(X_test)
#
# # Оценка качества модели
# rmse = mean_squared_error(y_test, predictions, squared=False)
#
# print("RMSE после оптимизации:", rmse)
#
# # Построение кривых валидации и обучения
# train_scores_mean = grid_search.cv_results_['mean_fit_time']
# train_scores_std = grid_search.cv_results_['std_fit_time']
# test_scores_mean = grid_search.cv_results_['mean_score_time']
# test_scores_std = grid_search.cv_results_['std_score_time']
#
# param_range = param_grid['poly_features__degree']
#
# plt.figure(figsize=(10, 6))
# plt.title('Validation Curve with Linear Regression')
# plt.xlabel('Polynomial Degree')
# plt.ylabel('Negative Mean Squared Error')
# plt.ylim(-0.5, 0)  # Устанавливаем пределы по оси y
# lw = 2
#
# plt.semilogx(param_range, -train_scores_mean, label='Training score', color='darkorange', lw=lw)
# plt.fill_between(param_range, -train_scores_mean - train_scores_std, -train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=lw)
#
# plt.semilogx(param_range, -test_scores_mean, label='Cross-validation score', color='navy', lw=lw)
# plt.fill_between(param_range, -test_scores_mean - test_scores_std, -test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=lw)
#
# plt.legend(loc='best')
# plt.show()
# две горизонтальные прямые на графике кривых валидации и обучения, это может указывать на то, что
# изменение гиперпараметра (в данном случае степени полинома) не влияет на качество модели
#


# оптимизация модели
from sklearn.ensemble import RandomForestRegressor

# Создание пайплайна для преобразования данных и обучения модели случайного леса
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('random_forest', RandomForestRegressor())
])

# Определение сетки параметров для подбора гиперпараметров
param_grid = {
    'poly_features__degree': [2, 3],  # Попробуем полиномиальные признаки второй и третьей степени
    'random_forest__n_estimators': [50, 100, 200],  # Количество деревьев в лесу
    'random_forest__max_depth': [None, 10, 20]  # Максимальная глубина деревьев
}

# Подбор гиперпараметров с использованием кросс-валидации
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Получение лучшей модели
best_model = grid_search.best_estimator_


# Сохраняем модель в файл
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)


# Прогнозирование на тестовой выборке
predictions = best_model.predict(X_test)

# Оценка качества модели
rmse = mean_squared_error(y_test, predictions, squared=False)

print("RMSE после оптимизации модели случайного леса:", rmse)


from sklearn.model_selection import validation_curve

# Определение диапазона значений параметра для построения кривых валидации
param_range = [50, 100, 200]

# Получение значений кривых валидации
train_scores, test_scores = validation_curve(
    estimator=best_model,
    X=X_train,
    y=y_train,
    param_name='random_forest__n_estimators',
    param_range=param_range,
    cv=5,
    scoring='neg_mean_squared_error'
)

# Расчет средних значений и стандартных отклонений для кривых валидации
train_scores_mean = np.mean(-train_scores, axis=1)
train_scores_std = np.std(-train_scores, axis=1)
test_scores_mean = np.mean(-test_scores, axis=1)
test_scores_std = np.std(-test_scores, axis=1)

# Построение кривых валидации
plt.figure(figsize=(10, 6))
plt.title('Validation Curve with Random Forest Regression')
plt.xlabel('Number of Estimators')
plt.ylabel('Negative Mean Squared Error')
plt.ylim(0, 0.5)  # Устанавливаем пределы по оси y
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score', color='darkorange', lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score', color='navy', lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=lw)

plt.legend(loc='best')
plt.show()

