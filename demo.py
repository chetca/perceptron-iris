import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

# Инициализация рандомки из пакета numpy
seed = 101
np.random.seed(seed)

# Грузим датасет используя пакет pandas
dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
dataset = dataframe.values

# Делим данные на две части – на целевую переменную Y и на данные, от которых она зависит X
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# Iris-setosa кодируется как класс 0
# Iris-versicolor – класс 1 
# Iris-virginica как класс 2 
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Затем применим унитарное кодирование к полученным категориями. 
# Iris-setosa станет [1, 0, 0],  
# Iris-versicolor – [0, 1, 0] и т.д. 
dummy_y = np_utils.to_categorical(encoded_Y)

# Отделяем часть данных для тестового подмножества для оценки качества работы нашей сети
# используя метод train_test_split
# В качестве параметров укажем размер тестового сета – 25% 
# и передадим наш инициализатор случайных значений
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=.25, random_state=seed)

# Описываем персептрон
# Используются методы из библиотека Keras
# Сеть следующей архитектуры: 
# 4 входа под каждый признак, 
# 4 нейрона в скрытом слое, 
# и три выхода — по количеству классов.
# функция активации - сигмоида
def my_model():
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Установим количество эпох обучения равное 200 
# и количество примеров для одновременного обучения равное 7
model = my_model()
model.fit(X_train, Y_train, batch_size=7, nb_epoch=200, verbose=1)

# Оценим качество нашей модели на тестовом подмножестве
# Будем использовать метрику площади под ROC-кривой
predictions = model.predict_proba(X_test)
print('Accuracy: {}'.format(roc_auc_score(y_true=Y_test, y_score=predictions)))

# Полезные ссылки: 
# (Ирисы Фишера) https://ru.wikipedia.org/wiki/%D0%98%D1%80%D0%B8%D1%81%D1%8B_%D0%A4%D0%B8%D1%88%D0%B5%D1%80%D0%B0
# (Унитарный код) https://ru.wikipedia.org/wiki/%D0%A3%D0%BD%D0%B8%D1%82%D0%B0%D1%80%D0%BD%D1%8B%D0%B9_%D0%BA%D0%BE%D0%B4
# (ROC-кривая) https://ru.wikipedia.org/wiki/ROC-%D0%BA%D1%80%D0%B8%D0%B2%D0%B0%D1%8F
# (Keras documentation) https://keras.io/
# (Данные) https://archive.ics.uci.edu/ml/datasets/iris