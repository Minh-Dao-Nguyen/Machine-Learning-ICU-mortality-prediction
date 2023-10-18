# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


from data import load_data, preprocess_x, split_data
from parser import parse
from model import Model
from sklearn.preprocessing import StandardScaler


def main():
    args = parse()

    x_train_path = 'train_x.csv'
    df_train = load_data(x_train_path)
    X = preprocess_x(df_train)

    y = pd.read_csv('train_y.csv', index_col=1)
    y = y.iloc[: , 1:]

    # combine x and y based on id
    df_model = pd.merge(X, y, left_index=True, right_index=True)

    X = df_model.iloc[: , : df_model.shape[1]-1]
    y = df_model.iloc[:,df_model.shape[1]-1]

    #standardize
    scaler = StandardScaler()
    # Fit the scaler to the data
    scaler.fit(X)
    # Transform the training and testing data
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    
    model = Model(args)  # you can add arguments as needed
    model.fit(X, y)

    y_pred = model.predict_proba(X_test)[:,1]
    rocauc = roc_auc_score(y_test, y_pred)
    print(rocauc)


if __name__ == "__main__":
    main()
