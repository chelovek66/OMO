import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Пример данных (замените на реальные)
data = {
    'Math': [85, 90, 78, 92, 70],
    'Russian': [80, 88, 82, 95, 75],
    'Physics': [70, 85, 65, 90, 60],
    'Admission_Score': [88, 92, 80, 95, 75]
}

df = pd.DataFrame(data)
X = df[['Math', 'Russian', 'Physics']].values  
y = df['Admission_Score'].values               

# Масштабируем признаки (важно для регуляризации)
X = (X - X.mean(axis=0)) / X.std(axis=0)
print(X)
# print(X)
def ridge_regression(X, y, lambda_):
    n_features = X.shape[1]
    I = np.eye(n_features)  # Единичная матрица
    XTX = X.T @ X
    XTy = X.T @ y
    w = np.linalg.inv(XTX + lambda_ * I) @ XTy  
    return w

# Диапазон λ (лучше в логарифмической шкале)
lambdas = np.logspace(-4, 4, 100)  # от 10^-4 до 10^4
weights = []

for l in lambdas:
    # print(l)
    w = ridge_regression(X, y, l)
    weights.append(w)

print(weig)

weights = np.array(weights)  # Массив размером (100, 3)



for i, subject in enumerate(['Math', 'Russian', 'Physics']):
    plt.plot(lambdas, weights[:, i], label=subject)

plt.xscale('log')  # Логарифмическая шкала для λ
plt.xlabel("λ (lambda, сила регуляризации)")
plt.ylabel("Вес признака")
plt.title("Зависимость весов признаков от λ (Ridge Regression)")
plt.legend()
plt.grid(True)
plt.show()