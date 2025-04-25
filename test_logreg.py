import numpy as np

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
    m, d = X.shape
    W = np.random.random((d, 1))
    losses = []
    for _ in range(iterations):
        preds = 1 / (1 + np.exp(-X @ W))
        loss = -(y * np.log(preds) + (1 - y) * np.log(1 - preds)).sum()
        losses.append(loss)
        dloss = ((1/m) * X.T @ (preds - y)).mean(axis=1)
        W -= dloss * learning_rate
        # TODO: fix this
    return W.tolist(), losses

def test():
    X = np.array([[1.0, 0.5], [-0.5, -1.5], [2.0, 1.5], [-2.0, -1.0]])
    y = np.array([1, 0, 1, 0])
    print(train_logreg(X, y, 0.01, 20))
    
test()