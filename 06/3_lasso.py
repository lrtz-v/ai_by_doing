import numpy as np
from sklearn.linear_model import Lasso

np.random.seed(10)
X = np.matrix(np.random.randint(5, size=(10, 10)))
Y = np.matrix(np.random.randint(10, size=(10, 1)))
alpha = 0.5

alphas = np.linspace(1, 10, 10)
lasso_coefs = []

for a in alphas:
    lasso = Lasso(alpha=a, fit_intercept=False)
    lasso.fit(np.asarray(X), np.asarray(Y))
    lasso_coefs.append(lasso.coef_)

print(lasso_coefs)
