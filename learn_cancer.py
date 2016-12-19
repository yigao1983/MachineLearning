import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.decomposition as dc
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.neural_network as nn
import sklearn.pipeline as pl
import matplotlib.pyplot as plt

def main():
    
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                     "breast-cancer-wisconsin/wdbc.data", header=None)
    X = df.iloc[:,2:].values
    y = pp.LabelEncoder().fit_transform(df.iloc[:,1].values)
    
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=1)
    
    pipe_lr = pl.Pipeline([("scl", pp.StandardScaler()), \
                           ("clf", lm.LogisticRegression(random_state=1))])
    
    #pipe_lr.fit(X_train, y_train)
    #print("Test Accuracy: {:.3f}".format(pipe_lr.score(X_test, y_test)))
    
    n_folds = 10
    kf = ms.KFold(n_splits=n_folds)
    
    scores = ms.cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=kf, n_jobs=1)
    print("CV accuracy: {:.3f} +/- {:.3f}".format(scores.mean(), scores.std(ddof=0)))
    
    train_sizes, train_scores, test_scores = ms.learning_curve(estimator=pipe_lr, X=X_train, y=y_train, \
                                                               train_sizes=np.linspace(0.1, 1.0, 10), cv=10)
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_mean, color="b", marker="o", label="train")
    plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.2, color="b")
    plt.plot(train_sizes, test_mean, color="g", marker="s", label="cv")
    plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.2, color="g")
    plt.grid()
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.legend(loc=4)
    plt.ylim(0.8, 1.0)
    plt.show()
    
    params = [.001, .01, .1, 1.0, 10.0, 100.0]
    
    train_scores, test_scores = ms.validation_curve(estimator=pipe_lr, X=X_train, y=y_train, \
                                                    param_name="clf__C", param_range=params, cv=kf)
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    plt.figure()
    plt.plot(params, train_mean, color="b", marker="o", label="train")
    plt.fill_between(params, train_mean+train_std, train_mean-train_std, alpha=0.2, color="b")
    plt.plot(params, test_mean, color="g", marker="s", label="cv")
    plt.fill_between(params, test_mean+test_std, test_mean-test_std, alpha=0.2, color="g")
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.legend(loc=4)
    plt.ylim(0.8, 1)
    plt.xscale('log')
    plt.show()
    
    #pipe_svc = pl.Pipeline([("scl", pp.StandardScaler()), \
    #                        ("clf", svm.SVC(random_state=1))])
    pipe_nn = pl.Pipeline([("scl", pp.StandardScaler()), \
                           ("clf", nn.MLPClassifier(random_state=1, max_iter=400, tol=1e-4, verbose=True))])
    
    layer_range = [(2,), (5,), (10,), (20,), (50,)]
    alpha_range = [.0001, .001, .01, .1, 1., 10., 100., 1000.]
    
    param_grid = {"clf__hidden_layer_sizes": layer_range, "clf__alpha": alpha_range}
    
    gs = ms.GridSearchCV(estimator=pipe_nn, param_grid=param_grid, scoring="accuracy", cv=kf, n_jobs=1)
    gs.fit(X_train, y_train)
    
    print(gs.best_score_)
    print(gs.best_params_)
    print(gs.best_estimator_.score(X_test, y_test))
    
if __name__ == "__main__":
    
    main()
