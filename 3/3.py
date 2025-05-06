import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = {
    'Математика': [85, 90, 78, 92, 70, 65, 74, 89],
    'Русский': [80, 88, 82, 95, 75, 78, 92, 70],
    'Физика': [70, 85, 65, 90, 60, 88, 92, 80],
    'Баллы': [264, 276, 240, 285, 225, 210, 195, 222]
}

d = pd.DataFrame(data)

x = d[['Математика','Русский','Физика']].values
y = d['Баллы'].values

x = (x - x.mean(axis=0)) / x.std(axis=0)
print(x)
# print(x)
def ridge_regression(x, y, lambda_):
    # print(x.shape[1])
    n_features = x.shape[1]
    I = np.eye(n_features)  # Единичная матрица
    XTX = x.T @ x
    XTy = x.T @ y
    w = np.linalg.inv(XTX + lambda_ * I) @ XTy # вычисление обратной матрицы 
    return w


lambdas = np.logspace(-4, 4, 100)
weights = []

for l in lambdas:
    w = ridge_regression(x, y, l)
    weights.append(w)


weights = np.array(weights)  


for i, subject in enumerate(['Математика', 'Русский', 'Физика']):
    plt.plot(lambdas, weights[:, i], label=subject)

plt.xscale('log') 
plt.xlabel("λ")
plt.ylabel("W")
plt.legend()
plt.grid(True)
plt.show()

