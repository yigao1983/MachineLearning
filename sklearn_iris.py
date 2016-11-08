import sklearn.datasets as datasets
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import numpy as np

def main():
    
    iris = datasets.load_iris()
    
    X = iris.data[:,[2,3]]
    y = iris.target
    
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)
    
    sc = pp.StandardScaler().fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

if __name__ == "__main__":
    
    main()
