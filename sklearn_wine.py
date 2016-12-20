import numpy as np
import pandas as pd

def main():
    
    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/"
                          "wine.data", header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', \
                       'Alcalinity of ash', 'Magnesium', 'Total phenols', \
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', \
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', \
                       'Proline']
    print(df_wine)

if __name__ == "__main__":
    
    main()
