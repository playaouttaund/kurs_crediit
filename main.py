import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # Импортируем StandardScaler для нормализации

# Загрузка данных
data = pd.read_csv('creditcard.csv')

# 1. Предобработка данных

# Проверка на наличие пропущенных значений
print("Проверка на пропущенные значения:")
print(data.isnull().sum())

# Удаление дубликатов
data = data.drop_duplicates()
print(f"Количество строк после удаления дубликатов: {data.shape[0]}")

# Визуализация: Количество мошеннических и легитимных транзакций
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Class', palette='Set2')
plt.title('Мошенничество против легитимных транзакций')
plt.xlabel('Класс (0: Легитимные, 1: Мошеннические)')
plt.ylabel('Количество')
plt.show()

# Визуализация: Распределение суммы транзакций по классам
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Amount', hue='Class', multiple='stack', palette='Set2', bins=50)
plt.title('Распределение суммы транзакций по классам')
plt.xlabel('Сумма транзакции')
plt.ylabel('Количество')
plt.show()

# Визуализация: Время vs Сумма транзакций
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Time', y='Amount', hue='Class', palette='Set1')
plt.title('Время против суммы транзакций')
plt.xlabel('Время (секунды)')
plt.ylabel('Сумма транзакции')
plt.show()

# Визуализация: Ящик с усами для суммы транзакций по классам
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='Class', y='Amount', palette='Set3')
plt.title('Ящик с усами для суммы транзакций по классам')
plt.xlabel('Класс')
plt.ylabel('Сумма транзакции')
plt.show()

# Визуализация: Мошеннические транзакции во времени
fraud = data[data['Class'] == 1]
fig = px.scatter(fraud, x='Time', y='Amount', title='Мошеннические транзакции во времени', labels={'Time':'Время (секунды)', 'Amount':'Сумма транзакции'})
fig.show()

# Визуализация: Распределение признака V1 по классам
plt.figure(figsize=(12, 6))
sns.violinplot(data=data, x='Class', y='V1', palette='Set2')
plt.title('Распределение признака V1 по классам')
plt.xlabel('Класс')
plt.ylabel('V1')
plt.show()

# Параметры легитимных и мошеннических транзакций
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
print("Легитимные транзакции:", legit.shape)
print("Мошеннические транзакции:", fraud.shape)

# Средние значения по классам
print(data.groupby('Class').mean())

# Случайная выборка легитимных транзакций
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Разделение данных на признаки и целевую переменную
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, Y_train)

# Оценка точности модели на обучающих данных
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Точность данных об обучении : ', training_data_accuracy)

# Оценка точности модели на тестовых данных
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Оценка точности тестовых данных : ', test_data_accuracy)
