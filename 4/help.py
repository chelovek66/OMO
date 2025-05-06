import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score

# data.info()


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

class Node:
    def __init__(self, index , t, true_branch, false_branch):
        self.index = index #индекс признака
        self.t = t#значение порога 
        self.true_branch = true_branch #поддерево да
        self.false_branch = false_branch#поддерево нет 

#класс листа
class Leaf:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels 
        self.prediction = np.mean(labels)
    
    # def predict(self):

    #     classes = {}
    #     for label in self.labels:
    #         if label not in classes:
    #             classes[label]=0
    #         classes[label]+=1
        
    #     return max(classes, key=classes.get)

# def variance(labels):
#     # Расчет дисперсии (критерий для регрессии)
#     if len(labels) == 0:
#         return 0
#     return np.var(labels)

def gini(labels):
    #  подсчет количества объектов разных классов
    classes = {}
    for label in labels:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1

    #  расчет критерия
    impurity = 1
    for label in classes:
        p = classes[label] / len(labels)
        impurity -= p ** 2

    return impurity

# Расчет прироста

def gain(left_labels, right_labels, root_variance):
    n = len(left_labels) + len(right_labels)
    p_left = len(left_labels) / n
    p_right = len(right_labels) / n
    return root_variance - (p_left * gini(left_labels) + p_right * gini(right_labels))

def split(data, labels, column_index, t):

    left = data[:,column_index]<=t
    right = ~left

    true_data = data[left]
    false_data = data[right]

    true_labels = labels[left]
    false_labels = labels[right]

    return true_data, false_data, true_labels, false_labels

def find_best_split(data, labels):
    min_samples_leaf = 5
    root_variance = gini(labels)
    best_gain = -np.inf
    best_t = None
    best_index = None

    for index in range(data.shape[1]):
        values = np.unique(data[:, index])
        
        for t in values:
            true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
            
            if len(true_labels) < min_samples_leaf or len(false_labels) < min_samples_leaf:
                continue
                
            current_gain = gain(true_labels, false_labels, root_variance)
            
            if current_gain > best_gain:
                best_gain, best_t, best_index = current_gain, t, index
                
    return best_gain, best_t, best_index

def build_tree(data, labels, depth =0 , max_depth=5, current_leaves=[0], max_leaves=10):
    gain, t, index = find_best_split(data,labels)

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
    
    true_data,false_data, true_labels, false_labels = split(data, labels, index, t)
    true_branch = build_tree(true_data,true_labels, depth+1, max_depth=max_depth,current_leaves=current_leaves, max_leaves=max_leaves)
    false_branch  = build_tree(false_data, false_labels, depth+1, max_depth=max_depth, current_leaves=current_leaves, max_leaves=max_leaves)

    return Node(index, t, true_branch, false_branch)
def classify_obj(obj, node):

    if isinstance(node,Leaf):
        answer = node.prediction
        return answer
    
    if obj[node.index]<=node.t:
        return classify_obj(obj, node.true_branch)
    else:
        return classify_obj(obj, node.false_branch)

def predict(data, tree):
    classes = []
    for obj in data:
        prediction = classify_obj(obj, tree)
        classes.append(prediction)

    return classes

# Разобьем выборку на обучающую и тестовую

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(classification_data,
                                                                    classification_labels,
                                                                    test_size=0.3,
                                                                    random_state=1)

my_tree = build_tree(train_data, train_labels, max_depth=3, max_leaves=3)


# Предсказания и метрики
train_pred = predict(train_data, my_tree)
test_pred = predict(test_data, my_tree)

def mse_metric(actual, predicted):
    return np.mean((actual - predicted)**2)

train_mse = mse_metric(train_labels, train_pred)
test_mse = mse_metric(test_labels, test_pred)

print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# # Получим ответы для обучающей выборки
# train_answers = predict(train_data, my_tree)

# # И получим ответы для тестовой выборки
# answers = predict(test_data, my_tree)

# # Введем функцию подсчета точности как доли правильных ответов
# def accuracy_metric(actual, predicted):
#     correct = 0
#     for i in range(len(actual)):
#         if actual[i] == predicted[i]:
#             correct += 1
#     return correct / float(len(actual)) * 100.0

# # Точность на обучающей выборке
# train_accuracy = accuracy_metric(train_labels, train_answers)
# print(train_accuracy)

# # Точность на тестовой выборке
# test_accuracy = accuracy_metric(test_labels, answers)
# print(test_accuracy)


# Визуализируем дерево на графике

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