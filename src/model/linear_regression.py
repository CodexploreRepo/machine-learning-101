import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            # Compute gradient
            # dJ/dw = (1/N)* sum[(X)*(y_pred-y)] = (1/N)[X dot (y_pred-y)]
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            # dJ/db = (1/N)*sum[(y_pred-y)]
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weight & bias
            self.weights -= self.lr * dw  # w = w - lr.dw
            self.bias -= self.lr * db  # b = b - lr.db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2024
    )
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)

    for lr in [0.001, 0.002, 1]:
        regressor = LinearRegression(lr=lr)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print(f"lr={lr:5}: mse={mean_squared_error(y_test, y_pred)}")

        plt.plot(X_test, y_pred, label=f"lr={lr}")
    plt.legend()
    plt.show()
