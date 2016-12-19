import numpy as np
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.pipeline as pl
import sklearn.linear_model as lm
import sklearn.svm as svm
import matplotlib.pyplot as plt

def main():
    
    n_sample = 10000
    n_feature = 2
    
    np.random.seed(1)
    
    X = np.random.randn(n_sample, n_feature)
    y = np.zeros(n_sample)
    
    w = np.array([2.0, 0.4])
    
    for i, xi in enumerate(X):
        y[i] = np.dot(w, xi) + 10.0*(np.sin(2*xi[0]))**3 + 5.0*np.cos(5*xi[1]) + np.random.normal(scale=2.0)
    
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2)
    
    scl = pp.StandardScaler()
    #reg = lm.LinearRegression(fit_intercept=True)
    reg = svm.SVR(kernel="rbf", C=1000, tol=1, verbose=True, max_iter=-1)
    
    pipe_lr = pl.Pipeline([("scl", scl), ("reg", reg)])
    
    pipe_lr.fit(X_train, y_train)
    
    y_pred = pipe_lr.predict(X_test)
    
    print(pipe_lr.score(X_test, y_test))
    
    plt.figure()
    plt.scatter(y_pred, y_test)
    plt.axis("equal")
    plt.xlim(y_test.min(), y_test.max())
    plt.ylim(y_test.min(), y_test.max())
    plt.show()
    
if __name__ == "__main__":
    
    main()
        
    
    
