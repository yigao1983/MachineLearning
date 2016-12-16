import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.decomposition as dc
import sklearn.linear_model as lm
import sklearn.pipeline as pl
import matplotlib.pyplot as plt

def main():
    
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                     "breast-cancer-wisconsin/wdbc.data", header=None)
    X = df.iloc[:,2:].values
    y = pp.LabelEncoder().fit_transform(df.iloc[:,1].values)
    
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=1)
    
    pipe_lr = pl.Pipeline([("scl", pp.StandardScaler()), \
                           ("pca", dc.PCA(n_components=2)), \
                           ("clf", lm.LogisticRegression(random_state=1))])
    
    n_folds = 10
    kf = ms.KFold(n_splits=n_folds)
    """
    scores = np.zeros(n_folds)
    
    for k, (idx_train, idx_cv) in enumerate(kf.split(X_train)):
        
        pipe_lr.fit(X_train[idx_train], y_train[idx_train])
        scores[k] = pipe_lr.score(X_train[idx_cv], y_train[idx_cv])
        
    print(pipe_lr)
    print(scores.mean(), scores.std(ddof=0))
    """
    #scores = ms.cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=kf, n_jobs=1)
    #print(scores)
    #print(scores.mean(), scores.std(ddof=0))
    
    train_sizes, train_scores, test_scores = ms.learning_curve(estimator=pipe_lr, X=X_train, y=y_train, \
                                                               train_sizes=np.linspace(0.1, 1.0, 5), cv=kf)
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_mean, color="b", marker="o")
    plt.show()
    
    
if __name__ == "__main__":
    
    main()
