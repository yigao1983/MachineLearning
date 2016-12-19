import numpy as np
import sklearn.svm as svm
import matplotlib.pyplot as plt
import sklearn_iris

def main():
    
    np.random.seed(0)
    
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    
    plt.figure()
    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c="b", marker="x", label=" 1")
    plt.scatter(X_xor[y_xor ==-1, 0], X_xor[y_xor ==-1, 1], c="r", marker="s", label="-1")
    plt.ylim(-3.0)
    plt.legend()
    plt.show()
    
    clf = svm.SVC(kernel="rbf", C=10.0, gamma=0.10, random_state=0)
    clf.fit(X_xor, y_xor)
    print(clf.score(X_xor, y_xor))
    
    plt.figure()
    sklearn_iris.plot_decision_regions(X_xor, y_xor, classifier=clf)
    plt.legend(loc=2)
    plt.show()
    
if __name__ == "__main__":
    
    main()
