import numpy as np
import pandas as pd
import io
import sklearn.preprocessing as pp

def main():
    
    csv_data = u'''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    0.1,11.0,12.0,'''
    
    df = pd.read_csv(io.StringIO(csv_data))
    print(df)
    """
    imr = pp.Imputer(missing_values="NaN", strategy="mean", axis=0).fit(df.values)
    
    imputed_data = imr.transform(df.values)
    
    #print(imputed_data)
    
    df = pd.DataFrame([['green', 'M', 10.1, 'class1'], \
                       ['red', 'L', 13.5, 'class2'], \
                       ['blue', 'XL', 15.3, 'class1']], \
                      columns=['color', 'size', 'price', 'classlabel'])
    #print(df)
    
    size_mapping = {'XL': 3, 'L': 2, 'M': 1}
    
    df['size'] = df['size'].map(size_mapping)
    
    #class_mapping = {label: idx for idx, label in enumerate(np.unique(df.classlabel))}
    #df['classlabel'] = df['classlabel'].map(class_mapping)
    
    X = pd.get_dummies(df[['color', 'size', 'price']]).values
    y = pp.LabelEncoder().fit_transform(df.classlabel.values)
    
    print(X)
    print(y)
    """
if __name__ == "__main__":
    
    main()
