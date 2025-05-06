import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap




class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
    
    def fit(self, X):
        # Центрирование данных
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # SVD разложение
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Выбор компонент
        if self.n_components is not None:
            Vt = Vt[:self.n_components]
            S = S[:self.n_components]
        
        self.components = Vt.T
        
        # Объясненная дисперсия
        total_var = np.sum(S**2) / (X.shape[0] - 1)
        self.explained_variance = (S**2) / (X.shape[0] - 1)
        self.explained_variance_ratio = self.explained_variance / total_var
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)





X, y = load_iris(return_X_y=True)

pca = PCA(n_components=2)
pca.fit(X) 
X_pca = pca.transform(X) 
# Для наглядности возьмем только первые два признака (всего в датасете их 4)
X = X[:, :2]




X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)
X_train.shape, X_test.shape

cmap = ListedColormap(['red', 'green', 'blue'])
plt.figure(figsize=(7, 7))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.show()


def e_metrics(x1, x2):

    distance = np.sum(np.square(x1 - x2))

    return np.sqrt(distance)


def knn(x_train, y_train, x_test, k):

    answers = []
    for x in x_test:
        test_distances = []

        for i in range(len(x_train)):

            # расчет расстояния от классифицируемого объекта до
            # объекта обучающей выборки
            distance = e_metrics(x, x_train[i])

            # Записываем в список значение расстояния и ответа на объекте обучающей выборки
            test_distances.append((distance, y_train[i]))

        # создаем словарь со всеми возможными классами
        classes = {class_item: 0 for class_item in set(y_train)}

        # Сортируем список и среди первых k элементов подсчитаем частоту появления разных классов
        for d in sorted(test_distances)[0:k]:
            classes[d[1]] += 1

        # Записываем в список ответов наиболее часто встречающийся класс
        answers.append(sorted(classes, key=classes.get)[-1])

    return answers
def weighted_knn(x_train, y_train, x_test, k):
    answers = []
    for x in x_test:
        distances = []
        for i in range(len(x_train)):
            distance = e_metrics(x, x_train[i])
            distances.append((distance, y_train[i]))
        
        # Сортировка по расстоянию и выбор k ближайших
        sorted_distances = sorted(distances, key=lambda x: x[0])[:k]
        
        # Взвешивание: вес = 1 / (расстояние + eps), чтобы избежать деления на 0
        eps = 1e-10  # Малое число для стабилизации
        weights = [1 / (d[0] + eps) for d in sorted_distances]
        classes = {class_item: 0 for class_item in set(y_train)}
        
        for (d, class_item), w in zip(sorted_distances, weights):
            classes[class_item] += w
        
        # Выбор класса с наибольшим суммарным весом
        answers.append(max(classes.items(), key=lambda x: x[1])[0])
    return answers

def accuracy(pred, y):
    return (sum(pred == y) / len(y))

def get_graph(X_train, y_train, k, ax, weighted=False):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
    h = .1

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Используем взвешенный или обычный kNN
    if weighted:
        Z = weighted_knn(X_train, y_train, np.c_[xx.ravel(), yy.ravel()], k)
    else:
        Z = knn(X_train, y_train, np.c_[xx.ravel(), yy.ravel()], k)
        
    Z = np.array(Z).reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(f"k = {k} ({'Weighted' if weighted else 'Standard'})")

# Сравнение стандартного и взвешенного kNN
plt.figure(figsize=(20, 8))

for k in range(1, 11):
    # Стандартный kNN
    ax = plt.subplot(2, 10, k)
    get_graph(X_train, y_train, k, ax, weighted=False)
    
    # Взвешенный kNN
    ax = plt.subplot(2, 10, k+10)
    get_graph(X_train, y_train, k, ax, weighted=True)

plt.tight_layout()
plt.show()

# Сравнение точности
standard_acc = []
weighted_acc = []

for k in range(1, 11):
    y_pred_std = knn(X_train, y_train, X_test, k)
    y_pred_w = weighted_knn(X_train, y_train, X_test, k)
    standard_acc.append(accuracy(y_pred_std, y_test))
    weighted_acc.append(accuracy(y_pred_w, y_test))
    print(f"k={k}: Knn={standard_acc[-1]:.3f}, Knn(с весом)={weighted_acc[-1]:.3f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1,11), standard_acc, label='Standard kNN', marker='o')
plt.plot(range(1,11), weighted_acc, label='Weighted kNN', marker='s')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()


def calculate_wcss(X, labels):
    """Вычисление среднего квадратичного внутрикластерного расстояния"""
    wcss = 0
    for cluster in set(labels):
        cluster_points = X[labels == cluster]
        centroid = np.mean(cluster_points, axis=0)
        wcss += np.sum((cluster_points - centroid)**2)
    return wcss / len(X)  # Среднее значение

wcss_std = []
wcss_weighted = []

for k in range(1, 11):
    # Для обучающей выборки
    y_pred_std = knn(X_train, y_train, X_train, k)
    y_pred_w = weighted_knn(X_train, y_train, X_train, k)
    
    wcss_std.append(calculate_wcss(X_train, y_pred_std))
    wcss_weighted.append(calculate_wcss(X_train, y_pred_w))

# График WCSS
plt.figure(figsize=(10,5))
plt.plot(range(1,11), wcss_std, label='Standard kNN', marker='o')
plt.plot(range(1,11), wcss_weighted, label='Weighted kNN', marker='s')
plt.xlabel('k')
plt.ylabel('WCSS (Within-Cluster SS)')
plt.title('Зависимость качества кластеризации от k')
plt.legend()
plt.grid()
plt.show()


print("Объясненная дисперсия по компонентам:", pca.explained_variance_ratio)
print("Суммарная объясненная дисперсия:", sum(pca.explained_variance_ratio))

cmap = ListedColormap(['red', 'green', 'blue'])
plt.figure(figsize=(7, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap)
plt.title("Данные Iris после PCA")
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.show()