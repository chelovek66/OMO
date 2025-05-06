import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
# from sklearn.preprocessing import StandardScaler


x, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
# x = StandardScaler().fit_transform(x)  # Нормализация данных
y = y.reshape(-1, 1)


X_b = np.c_[np.ones((len(x), 1)), x]

#Градиентный спуск
def gradient_descent(x, y, learning_rate, epochs):
    n_samples = len(x)
    weights = np.zeros((x.shape[1], 1))
    mse_history = []
    
    for epoch in range(epochs):
        y_pred = x @ weights
        error = y_pred - y
        gradient = (x.T @ error) / n_samples
        weights -= learning_rate * gradient
        mse = np.mean(error ** 2)
        mse_history.append(mse)
    
    return weights, mse_history

#Стохастический градиентный спуск
def stochastic_gradient_descent(x, y, learning_rate, epochs, batch_size):
    n_samples = len(x)
    weights = np.zeros((x.shape[1], 1))
    mse_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = x[indices]
        y_shuffled = y[indices]
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            
            y_pred = X_batch @ weights
            error = y_pred - y_batch
            gradient = (X_batch.T @ error) / len(X_batch)
            weights -= learning_rate * gradient
        
        mse = np.mean((x @ weights - y) ** 2)
        mse_history.append(mse)
    
    return weights, mse_history


gd_weights, gd_mse = gradient_descent(X_b, y, learning_rate=0.1, epochs=100)
sgd_weights, sgd_mse = stochastic_gradient_descent(X_b, y, learning_rate=0.01, epochs=100, batch_size=32)


plt.plot(gd_mse, label='Градиентный спуск (GD)', linewidth=2)
plt.plot(sgd_mse, label='Стохастический GD (SGD)', linewidth=2)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


print("Веса GD:", gd_weights.flatten())
print("Веса SGD:", sgd_weights.flatten())