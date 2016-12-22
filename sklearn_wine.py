import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.neighbors as nb
import sklearn.ensemble as es
import matplotlib.pyplot as plt
import SBS

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
    
if __name__ == "__main__":
    
    main()
