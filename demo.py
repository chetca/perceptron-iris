import numpy as np
import time


# формирование сети 4-4-3 для многослойного персептрона
n_hidden = 4
n_in = 4
n_out = 3

learning_rate = 0.01
momentum = 0.9

np.random.seed(0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def train(x, t, V, W, bv, bw):

    # счетчик ввода с весом матрицы скрытого слоя 1x4 - 4x4 дает матрицу 1x4
    A = np.dot(x, V) + bv
    Z = sigmoid(A)

   
    # рассчитатываем скрытый слой с весом матрицы выходного слоя 1x4 - 4x3 
    # для генерации матрицы 1x3
    B = np.dot(Z, W) + bw
    Y = sigmoid(B)
    

    # метод обратоного распространения ошибок для предыдущего слоя
    # ошибка на выходном уровне
    Ew = Y - t
    # ошибка в скрытом слое A является матрицей 4.1, умноженной на W = 4x3 Ew = 3.1
    # print "x"
    # print sigmoid(A)
    # print np.shape([4,5,6])
    # print "x"
    Ev = sigmoid(A) * np.dot(W, Ew)

    # ошибка для выходного слоя в точке внешнего с результатом скрытого слоя
    print("Z")
    print(Z)
    np.savetxt('out.txt', Z, delimiter=',')
    print("Z")
    print("Ew")
    print(Ew)
    np.savetxt('out.txt', Ew, delimiter=',')
    print("Ew")
    dW = np.outer(Z, Ew)
    # ошибка для скрытого слоя в точке внешнего с результатом входного слоя
    dV = np.outer(x, Ev)

    # Дополнительная формула потери перекрестной энтропии не использовалась в классе aia bu
    # loss = -np.mean ( t * np.log(Y) + (1 - t) * np.log(1 - Y) )

    # евклидово расстояние между результатами и фактами E (y, y ') из википедии
    loss =0.5*(np.sum(np.power(Y-t,2)))


    return  loss, (dV, dW, Ev, Ew)

def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    hasil = (sigmoid(B) > 0.5).astype(int)
    if hasil[0]==1:
        return 'Iris-setosa'
    elif hasil[1]==1:
        return 'Iris-versicolor'
    elif hasil[2]==1:
        return 'Iris-virginica'
    else:
        return 'Unknown'

# Настройка начальных параметров персептрона
# вес скрытых слоёв
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
# вес выходных слоёв
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

print(V)
print(W)

np.savetxt('out.txt', V, delimiter=',')
np.savetxt('out.txt', W, delimiter=',')

# смещение
bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V,W,bv,bw]

# разделение 120 итераций обучения на 30 тестирований
datatraining = np.loadtxt('iris.data-feature.txt',delimiter=',')
datalabel = np.loadtxt('iris.data-feature-label.txt',delimiter=',')

# вывод данных по тренировке сети
# Собственно сама тренировка
for epoch in range(10000):
    err = []
    upd = [0]*len(params)

    t0 = time.clock()
    for i in range(datatraining.shape[0]):
        loss, grad = train(datatraining[i], datalabel[i], *params)

        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learning_rate * grad[j]

        err.append( loss )


    
    print("Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 ))

# Результаты модели
print("FORECASTING DATA")
# print datatest
# print datalabeltest
# print params
for i in range(datatest.shape[0]):
    print(predict(datatest[i], *params))

