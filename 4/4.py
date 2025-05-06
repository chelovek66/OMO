import matplotlib.pyplot as plt
import random
import time
from math import log2

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_circles
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')



# сгенерируем данные
classification_data, classification_labels = make_classification(n_features=2, n_informative=2,
                                                                 n_classes=2, n_redundant=0,
                                                                 n_clusters_per_class=1, random_state=5)
# classification_data, classification_labels = make_circles(n_samples=30, random_state=5)

# визуализируем сгенерированные данные
colors = ListedColormap(['red', 'blue'])
light_colors = ListedColormap(['lightcoral', 'lightblue'])

plt.figure(figsize=(6,6))
plt.scatter(list(map(lambda x: x[0], classification_data)), list(map(lambda x: x[1], classification_data)),
              c=classification_labels, cmap=colors)
plt.show()


# Реализуем класс узла
class Node:

    def __init__(self, index, t, true_branch, false_branch):
        self.index = index  # индекс признака, по которому ведется сравнение с порогом в этом узле
        self.t = t  # значение порога
        self.true_branch = true_branch  # поддерево, удовлетворяющее условию в узле
        self.false_branch = false_branch  # поддерево, не удовлетворяющее условию в узле


# И класс терминального узла (листа)
class Leaf:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.prediction = np.mean(labels)

    # def predict(self):
    #     # подсчет количества объектов разных классов
    #     classes = {}  # сформируем словарь "класс: количество объектов"
    #     for label in self.labels:
    #         if label not in classes:
    #             classes[label] = 0
    #         classes[label] += 1

    #     # найдем класс, количество объектов которого будет максимальным в этом листе и вернем его
    #     prediction = max(classes, key=classes.get)
    #     return prediction


# Расчет критерия Джини
def dispersia(labels):
    #  подсчет количества объектов разных классов
    classes = {}
    for label in labels:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1

    #  расчет критерия
    impurity = 0
    for label in classes:
        p = classes[label] / len(labels)
        impurity -= p * log2(p)

    return impurity


# Расчет прироста
def gain(left_labels, right_labels, root_gini):

    # доля выборки, ушедшая в левое поддерево
    p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])

    return root_gini - p * dispersia(left_labels) - (1 - p) * dispersia(right_labels)


# Разбиение датасета в узле
def split(data, labels, column_index, t):

    left = np.where(data[:, column_index] <= t)
    right = np.where(data[:, column_index] > t)

    true_data = data[left]
    false_data = data[right]

    true_labels = labels[left]
    false_labels = labels[right]

    return true_data, false_data, true_labels, false_labels


# Нахождение наилучшего разбиения
def find_best_split(data, labels):
    #  обозначим минимальное количество объектов в узле
    min_samples_leaf = 3

    root_gini = dispersia(labels)

    best_gain = 0
    best_t = None
    best_index = None

    n_features = data.shape[1]

    for index in range(n_features):
        # будем проверять только уникальные значения признака, исключая повторения
        t_values = np.unique(data[:, index])

        for t in t_values:
            true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
            #  пропускаем разбиения, в которых в узле остается менее 5 объектов
            if len(true_data) < min_samples_leaf or len(false_data) < min_samples_leaf:
               continue

            current_gain = gain(true_labels, false_labels, root_gini)

            #  выбираем порог, на котором получается максимальный прирост качества
            if current_gain > best_gain:
                best_gain, best_t, best_index = current_gain, t, index

    return best_gain, best_t, best_index


# Построение дерева с помощью рекурсивной функции
def build_tree(data, labels, depth = 0, max_depth = 5, current_leaves = [0], max_leaves = 10):

    gain, t, index = find_best_split(data, labels)

    if depth >= max_depth:
        current_leaves[0]+=1
        return Leaf(data,labels)

    if current_leaves[0] >=max_leaves:
        return Leaf(data,labels)
    
    if len(np.unique(labels)) == 1:
        current_leaves[0] += 1
        return Leaf(data, labels)

    if gain<=0.01:
        current_leaves[0]+=1
        return Leaf(data, labels)


    true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
    true_branch = build_tree(true_data,true_labels, depth+1, max_depth=max_depth,current_leaves=current_leaves, max_leaves=max_leaves)
    false_branch  = build_tree(false_data, false_labels, depth+1, max_depth=max_depth, current_leaves=current_leaves, max_leaves=max_leaves)


    return Node(index, t, true_branch, false_branch)


def classify_object(obj, node):

    #  Останавливаем рекурсию, если достигли листа
    if isinstance(node, Leaf):
        answer = node.prediction
        return answer

    if obj[node.index] <= node.t:
        return classify_object(obj, node.true_branch)
    else:
        return classify_object(obj, node.false_branch)
    
def predict(data, tree):

    classes = []
    for obj in data:
        prediction = classify_object(obj, tree)
        classes.append(prediction)
    return classes


# Разобьем выборку на обучающую и тестовую
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(classification_data,
                                                                    classification_labels,
                                                                    test_size=0.3,
                                                                    random_state=1)


# Построим дерево по обучающей выборке
my_tree = build_tree(train_data, train_labels, max_depth = 3, max_leaves = 3)


# Предсказания и метрики
train_pred = predict(train_data, my_tree)
test_pred = predict(test_data, my_tree)

def mse_metric(actual, predicted):
    return np.mean((actual - predicted)**2)

train_mse = mse_metric(train_labels, train_pred)
test_mse = mse_metric(test_labels, test_pred)

print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

def get_meshgrid(data, step=.05, border=1.2):
    x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
    y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))


def visualize(train_data, test_data):
    plt.figure(figsize = (10, 5))

    # график обучающей выборки
    plt.subplot(1,2,1)
    xx, yy = get_meshgrid(train_data)
    mesh_predictions = np.array(predict(np.c_[xx.ravel(), yy.ravel()], my_tree)).reshape(xx.shape)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
    plt.scatter(train_data[:, 0], train_data[:, 1], c = train_labels, cmap = colors)
    plt.title(f'Train mse {train_mse}')

    # график тестовой выборки
    plt.subplot(1,2,2)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
    plt.scatter(test_data[:, 0], test_data[:, 1], c = test_labels, cmap = colors)
    plt.title(f'Test mse {test_mse}')
    plt.show()

visualize(train_data, test_data)