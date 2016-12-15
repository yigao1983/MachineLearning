import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as ms
import sklearn.decomposition as dc
import sklearn.linear_model as lm
import sklearn.pipeline as pipeline

def main():
    
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                     "breast-cancer-wisconsin/wdbc.data", header=None)
    X = df.iloc[:,2:].values
    y = preprocessing.LabelEncoder().fit_transform(df.iloc[:,1].values)
    
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=1)
    
    pipe_lr = pipeline.Pipeline([("scl", preprocessing.StandardScaler()), \
                                 ("pca", dc.PCA(n_components=2)), \
                                 ("clf", lm.LogisticRegression(random_state=1))])
    
    pipe_lr.fit(X_train, y_train)
    
    print("accuracy: {}".format(pipe_lr.score(X_test, y_test)))
    
if __name__ == "__main__":
    
    main()
