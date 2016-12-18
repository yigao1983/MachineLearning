import sklearn.datasets as datasets
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.linear_model as lm
import sklearn.svm as svm
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
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8, \
                    c=cmap(idx), marker=markers[idx], label=cl)
    
    # highlight test samples
    if not test_idx is None:
        X_test, y_test = X[test_idx,:], y[test_idx]
        plt.scatter(x=X_test[:,0], y=X_test[:,1], alpha=1.0, \
                    c="", linewidth=1, marker="o", s=55, label="test set")
    
def main():
    
    iris = datasets.load_iris()
    
    X = iris.data[:,[2,3]]
    y = iris.target
    
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)
    
    sc = pp.StandardScaler().fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    #ppn = lm.Perceptron(n_iter=40, eta0=0.01, random_state=0).fit(X_train_std, y_train)
    #y_pred = ppn.predict(X_test_std)
    
    #lr = lm.LogisticRegression(C=1000, random_state=0).fit(X_train_std, y_train)
    #y_pred = lr.predict(X_test_std)
    
    clf = svm.SVC(kernel="linear", C=1.0, random_state=0).fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    
    print('Misclassified samples: {}'.format((y_test != y_pred).sum()))
    print('Accuracy: {:10.2f}'.format(metrics.accuracy_score(y_test, y_pred)))
    
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    plt.figure()
    #plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
    #plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105, 150))
    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=clf, test_idx=range(105, 150))
    plt.xlabel('petal length (standardized)')
    plt.ylabel('petal width (standardized)')
    plt.legend(loc=4)
    plt.show()
    
    weights, params = [], []
    
    for c in np.arange(-5, 5):
        #lr = lm.LogisticRegression(C=10**c, random_state=0)
        #lr.fit(X_train_std, y_train)
        #weights.append(lr.coef_[1])
        clf = svm.SVC(kernel="linear", C=10**c, random_state=0)
        clf.fit(X_train_std, y_train)
        weights.append(clf.coef_[1])
        params.append(10**c)
    
    weights = np.array(weights)
    
    plt.figure()
    plt.plot(params, weights[:, 0], label='petal length')
    plt.plot(params, weights[:, 1], label='petal width', linestyle="--")
    plt.xlabel("C")
    plt.ylabel("weight coefficient")
    plt.xscale("log")
    plt.show()

if __name__ == "__main__":
    
    main()
