import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.decomposition as dc
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import SBS

def plot_decision_regions(X, y, classifier, resolution=0.02):
    
    markers = ("s", "x", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = clrs.ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), \
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="best")
    plt.show()
    
def main():
    
    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/"
                          "wine.data", header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', \
                       'Alcalinity of ash', 'Magnesium', 'Total phenols', \
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', \
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', \
                       'Proline']
    
    X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
    
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)
    
    stdsc = pp.StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    
    pca = dc.PCA(n_components=2)
    lr = lm.LogisticRegression()
    
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    """
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    plt.figure()
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=l, marker=m)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend("best")
    plt.show()
    """
    lr.fit(X_train_pca, y_train)
    
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    
    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    """
    plt.figure()
    plt.bar(range(1, 14), var_exp, alpha=0.5, align="center", label="individual explained variance")
    plt.step(range(1, 14), cum_var_exp, where="mid", label="cumulative explained variance")
    plt.xlabel("Principal components")
    plt.ylabel("Explained variance ratio")
    plt.legend(loc="best")
    plt.show()
    
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)
    
    w = np.vstack((eigen_pairs[0][1], eigen_pairs[1][1])).T
    
    X_train_pca = X_train_std.dot(w)
    
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    
    """
    
    """
    knn = nb.KNeighborsClassifier(n_neighbors=2)
    sbs = SBS.SBS(estimator=knn, k_features=1).fit(X_train_std, y_train)
    
    k_feat = [len(k) for k in sbs.subsets_]
    
    plt.figure()
    plt.plot(k_feat, sbs.scores_, marker="o")
    plt.ylim(0.7, 1.1)
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.show()
    
    feat_labels = df_wine.columns[1:]
    forest = es.RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print("{:2d}) {:<} {:f}".format(f+1, feat_labels[f], importances[indices[f]]))
    
    plt.figure()
    plt.title("Feature Importancese")
    plt.bar(range(X_train.shape[1]), importances[indices], color="lightblue", align="center")
    plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()
    """
if __name__ == "__main__":
    
    main()
    
