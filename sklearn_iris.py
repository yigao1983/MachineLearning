import sklearn.datasets as datasets
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import sklearn.pipeline as pl
import sklearn.metrics as metrics
import numpy as np
import matplotlib.colors as clrs
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    
    cmap = clrs.ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), \
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx, :]
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8, \
                    c=cmap(idx), marker=markers[idx], label=cl)
    
def main():
    
    iris = datasets.load_iris()
    
    X = iris.data[:,[2,3]]
    y = iris.target
    
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)
    
    sc = pp.StandardScaler().fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    ppn = lm.Perceptron(n_iter=40, eta0=0.01, random_state=0).fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    
    print('Misclassified samples: {}'.format((y_test != y_pred).sum()))
    print('Accuracy: {:10.2f}'.format(metrics.accuracy_score(y_test, y_pred)))
    
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    plt.figure()
    plot_decision_regions(X_test_std, y_test, ppn)
    plt.xlabel('petal length (standardized)')
    plt.ylabel('petal width (standardized)')
    plt.show()
    
    scl = pp.StandardScaler()
    clf = nn.MLPClassifier(max_iter=500, tol=1e-4, random_state=1, verbose=True)
    
    pipe_nn = pl.Pipeline([("scl", scl), ("clf", clf)])
    
    kf = ms.KFold(n_splits=10)
    
    layer_range = [(2,), (5,), (10,), (20,)]
    alpha_range = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    
    param_grid = {"clf__hidden_layer_sizes": layer_range, "clf__alpha": alpha_range}
    
    gs = ms.GridSearchCV(estimator=pipe_nn, param_grid=param_grid, scoring="accuracy", cv=kf)
    gs.fit(X_train, y_train)
    
    print(gs.best_params_)
    print(gs.best_score_)
    
    plt.figure()
    plot_decision_regions(X_test, y_test, gs.best_estimator_)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.show()  
    
if __name__ == "__main__":
    
    main()
