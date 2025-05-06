import matplotlib.pyplot as plt
import random
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

classification_data, classification_labels = make_classification(n_samples=1000,
                                                                 n_features=2, n_informative=2,
                                                                 n_classes=2, n_redundant=0,
                                                                 n_clusters_per_class=1, random_state=23)


colors = ListedColormap(['red', 'blue'])
light_colors = ListedColormap(['lightcoral', 'lightblue'])

plt.figure(figsize=(8,8))
plt.scatter(classification_data[:, 0], classification_data[:, 1],
              c=classification_labels, cmap=colors)
plt.show()



np.random.seed(42)
# делаем бустрэп. на вход передаем данные: признаки, целевые значения и кол-во подвыборок
# сколько мы хотим получить на выходе подвыборок
def get_bootstrap(data, labels, N):
    #получаем индексы объектов
    n_samples = data.shape[0] # размер совпадает с исходной выборкой
    # делаем заготовку для будущих бустрапированных подвыборок
    bootstrap = []

    # проходимся по количеству будущих деревьев нашей композиции
    for i in range(N):

        #генерируем индексы в том количестве сколько объектов у нас было
        #сколько объектов будет в бустрапированной подвыборке
        sample_index = np.random.randint(0, n_samples, size=n_samples)
        #признаки для обучения
        b_data = data[sample_index]
        #получем целевые значения для обучения
        b_labels = labels[sample_index]

        #добавляем в бустрапированную выборк
        #на выходе получаем кортеж из признаков и целевых значений
        bootstrap.append((b_data, b_labels))

    return bootstrap


#метод случайных подпространств
#получение признаков для каждого уникальног вопроса
#на вход подаем количество признаков
def get_subsample(len_sample):
    # будем сохранять не сами признаки, а их индексы
    sample_indexes = list(range(len_sample))

    #берем то количество признаков, которое нам рекомендовано
    #для классификации по формуле
    len_subsample = int(np.round(np.sqrt(len_sample)))

    # берем признаки без повторения, т.к. replace=False
    subsample = np.random.choice(sample_indexes, size=len_subsample, replace=False)

    return subsample

# Реализуем класс узла, где лежит сам вопрос
class Node:

    def __init__(self, index, t, true_branch, false_branch):
        self.index = index  # индекс признака, по которому ведется сравнение с порогом в этом узле
        self.t = t  # значение порога
        self.true_branch = true_branch  # ссылка на поддерево, удовлетворяющее условию в узле
        self.false_branch = false_branch  # ссылка на поддерево, не удовлетворяющее условию в узле\


# И класс терминального узла (листа)
class Leaf:

    def __init__(self, data, labels):
        self.data = data #данные
        self.labels = labels #целевые значения
        self.prediction = self.predict() #получаем предсказания

    def predict(self):#считаем предстказание
        # подсчет количества объектов разных классов, сколько каждого класса у нас появляется
        classes = {}  # сформируем словарь "класс: количество объектов"
        for label in self.labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1

        # найдем класс, количество объектов которого будет максимальным в этом листе и вернем его
        # берем тот класс, который появляется чаще, устраиваем голосование
        prediction = max(classes, key=classes.get)
        return prediction
    

# Расчет критерия Джини
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
        # используем вероятности появления класса в каждой подвыборке
        p = classes[label] / len(labels)
        # вычитаем из 1
        impurity -= p ** 2

    return impurity


# Расчет  информации
def gain(left_labels, right_labels, root_gini):

    # доля выборки, ушедшая в левое поддерево
    #нормируем критерий информативности по подвыборкам
    p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])
    #перемножаем на критерий Джини какая доля объектов ушла в левую подвыборку, какая доля ушла в правую
    # вычитаем это из критерия Джини вершины
    #т.е. сколько полезной информации добавилось в зависимоти от этого вопроса
    return root_gini - p * gini(left_labels) - (1 - p) * gini(right_labels)


# Разбиение датасета в узле
def split(data, labels, column_index, t):

    #разбиваем данные с помощью вопроса
    #получаем индексы где вопрос удовлетворяется
    left = np.where(data[:, column_index] <= t)
    #и где не удовлетворяется
    right = np.where(data[:, column_index] > t)

    #разбили на левую и правлую подвыборки по признакам
    true_data = data[left]
    false_data = data[right]

    # и по целевым значениям
    true_labels = labels[left]
    false_labels = labels[right]

    return true_data, false_data, true_labels, false_labels


# Нахождение наилучшего разбиения
def find_best_split(data, labels):

    #  обозначим минимальное количество объектов в узле
    #min_leaf_samples = 5

    #посчитали критерий Джини в вершине
    root_gini = gini(labels)

    #обнулили прирост
    best_gain = 0
    #нет лучшего порогового значения
    best_t = None
    #нет лучшего признака
    best_index = None

    #получили количество признаков
    n_features = data.shape[1]

    #генерируем случайные признаки на которых будем выбирать самый лучший вопрос
    feature_subsample_indices = get_subsample(n_features) # выбираем случайные признаки

    #проходимя по признакам
    for index in feature_subsample_indices:
        # будем проверять только уникальные значения признака, исключая повторения
        t_values = np.unique(data[:, index])

        #проходимся по уникальным значениям
        for t in t_values:
            true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
            #  пропускаем разбиения, в которых в узле остается менее 5 объектов
#             if len(true_data) < min_leaf_samples or len(false_data) < min_leaf_samples:
#                 continue

            current_gain = gain(true_labels, false_labels, root_gini)

            #  выбираем порог, на котором получается максимальный прирост качества
            #если прирост информации оказался лучше чем был, то все характеристики перезапиываем
            if current_gain > best_gain:
                best_gain, best_t, best_index = current_gain, t, index

    return best_gain, best_t, best_index


# Построение дерева с помощью рекурсивной функции
def build_tree(data, labels):

    gain, t, index = find_best_split(data, labels)

    #  Базовый случай - прекращаем рекурсию, когда нет прироста в качества
    if gain == 0:
      #возвращаем лист
        return Leaf(data, labels)

    #если прирост есть, продолжаем разбивать подвыборку
    true_data, false_data, true_labels, false_labels = split(data, labels, index, t)

    # Рекурсивно строим два поддерева
    true_branch = build_tree(true_data, true_labels)
    false_branch = build_tree(false_data, false_labels)

    # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева
    return Node(index, t, true_branch, false_branch)


#строим случайный лес
#на вход передаем данные, целевые значения и количнство деревьев
def random_forest(data, labels, n_trees):
    forest = []
    oob_predictions = {}  # Словарь для хранения OOB предсказаний
    
    for i in range(n_trees):
        # Генерируем бутстрап выборку
        n_samples = data.shape[0]
        sample_indices = np.random.randint(0, n_samples, size=n_samples)
        oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(sample_indices))
        
        b_data = data[sample_indices]
        b_labels = labels[sample_indices]
        
        # Строим дерево
        tree = build_tree(b_data, b_labels)
        forest.append(tree)
        
        # Делаем предсказания для OOB объектов
        if len(oob_indices) > 0:
            oob_data = data[oob_indices]
            predictions = predict(oob_data, tree)
            
            for idx, pred in zip(oob_indices, predictions):
                if idx not in oob_predictions:
                    oob_predictions[idx] = []
                oob_predictions[idx].append(pred)
    # Вычисляем OOB accuracy
    oob_accuracy = None
    if oob_predictions:
        actual = []
        voted = []
        for idx in oob_predictions:
            actual.append(labels[idx])
            voted.append(max(set(oob_predictions[idx]), key=oob_predictions[idx].count))
        oob_accuracy = accuracy_metric(actual, voted)
    
    return forest, oob_accuracy


# Функция классификации отдельного объекта
#на вход потупает объект и узел
def classify_object(obj, node):

    #  Останавливаем рекурсию, если достигли листа
    #если узел это лист, возвращаем предсказание
    if isinstance(node, Leaf):
        answer = node.prediction
        return answer

    #если узел не листок, сравниваем его с пороговым значением
    #если условие удовлетворяется, то рекурсовно идем на классификацию объекта
    if obj[node.index] <= node.t:
        return classify_object(obj, node.true_branch)
    else:
        return classify_object(obj, node.false_branch)
    

# функция формирования предсказания по выборке на одном дереве

def predict(data, tree):

    #формируем список предсказаний для дерева
    classes = []
    for obj in data:
        #классифицируем объект и добавляем предсказание в список
        prediction = classify_object(obj, tree)
        classes.append(prediction)
    return classes


# предсказание голосованием деревьев
#на вход поступает список деревьев и данные
def tree_vote(forest, data):

    # добавим предсказания всех деревьев в список
    predictions = []
    #проходимся по все деревьям, получаем дерево и предстказание по нему
    for tree in forest:
        predictions.append(predict(data, tree))
    # print(predictions)

    # сформируем список с предсказаниями для каждого объекта
    predictions_per_object = list(zip(*predictions))
    # print(predictions_per_object)

    # выберем в качестве итогового предсказания для каждого объекта то,
    # за которое проголосовало большинство деревьев
    voted_predictions = []
    for obj in predictions_per_object:
        voted_predictions.append(max(set(obj), key=obj.count))

    return voted_predictions


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


train_data, test_data, train_labels, test_labels = train_test_split(classification_data,
                                                                    classification_labels,
                                                                    test_size=0.3,
                                                                    random_state=1)



def get_meshgrid(data, step=.05, border=1.2):
    x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
    y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))


List = [1, 3, 10, 50]

for i in List:
    # Обучаем лес с OOB оценкой
    a, oob_acc = random_forest(classification_data, classification_labels, i)
    
    # Традиционная оценка на обучающей выборке (для сравнения)
    train_answers = tree_vote(a, classification_data)
    train_accuracy = accuracy_metric(classification_labels, train_answers)
    
    print(f'Случайный лес из {i} деревьев:')
    print(f'OOB точность: {oob_acc:.3f}' if oob_acc is not None else 'OOB оценка недоступна (нет OOB объектов)')
    print(f'Точность на всей выборке: {train_accuracy:.3f}')
    
    # Визуализация
    xx, yy = get_meshgrid(classification_data)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_predictions = np.array(tree_vote(a, grid_points)).reshape(xx.shape)
    
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors)
    plt.scatter(classification_data[:, 0], classification_data[:, 1], 
                c=classification_labels, cmap=colors)
    plt.title(f'Границы решений ({i} деревьев)\nOOB accuracy: {oob_acc:.2f}%' if oob_acc else f'Границы решений ({i} деревьев)')
    plt.show()
