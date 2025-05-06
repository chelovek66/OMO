import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gradient_descent_l1(x, y, learning_rate=0.01, lambda_=0.1, epochs=1000, tolerance=1e-4):
    n_samples, n_features = x.shape
    weights = np.zeros(n_features)
    history = []
    
    for epoch in range(epochs):
        y_pred = x @ weights
        error = y_pred - y
        
        gradient_mse = (x.T @ error) / n_samples
        gradient_l1 = lambda_ * np.sign(weights) 
        gradient = gradient_mse + gradient_l1
        

        new_weights = weights - learning_rate * gradient
        

        if np.linalg.norm(new_weights - weights) < tolerance:
            break
            
        weights = new_weights
        
        mse = np.mean(error ** 2)
        l1_penalty = lambda_ * np.sum(np.abs(weights))
        total_loss = mse + l1_penalty
        history.append(total_loss)
    
    return weights, history


# Синтетические данные
np.random.seed(42)
x = np.random.randn(100, 5)
y = 3 * x[:, 0] - 1.5 * x[:, 1] + 0.5 * x[:, 2] + np.random.randn(100) * 0.5

X_b = np.c_[np.ones((100, 1)), x]

weights, history = gradient_descent_l1(X_b, y, lambda_=0.5)

print("Веса модели:", weights)


plt.plot(history)
plt.xlabel("Итерации")
plt.ylabel("Функция потерь (MSE + L1)")
plt.title("Градиентный спуск с L1-регуляризацией")
plt.grid()
plt.show()