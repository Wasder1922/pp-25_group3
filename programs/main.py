import pandas as pd
import sklearn as sk
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# чтение данных
df = pd.read_csv('whole_data.csv')

# удаление колонок
delcols = [
    'type',
    'Bmag',
    # 'gpmag', # высокая корреляция
    # 'rpmag', # высокая корреляция
    # 'ipmag', # высокая корреляция
    # 'err',
    # 'nobs',
    'e_ipmag',
    'e_Vmag',
    'e_rpmag',
    'e_gpmag',
    'e_Bmag'
]
data = df.drop(columns=delcols)

# удаление NaN-ов
data = data.dropna().reset_index()

# стандатизируем признаки
scaler = StandardScaler()
present = data['present']
newdata = pd.DataFrame(data = scaler.fit_transform(data),
 columns = data.columns)

# уменьшаем кол-во измерений
pca = PCA(n_components=6)
pca = pca.fit_transform(newdata)
pca_df = pd.DataFrame(data = pca)
pca_df['present'] = present

# разделение выборки на тренировочную и тестовую
y = pca_df['present']
X = pca_df.drop(columns='present')
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, train_size=0.8)

# тренировка модели
classifier = sk.ensemble.RandomForestClassifier()
classifier.fit(X_train, y_train)

# проверка метрикой f1
res = classifier.predict(X_test)
sk.metrics.f1_score(y_test, res)
