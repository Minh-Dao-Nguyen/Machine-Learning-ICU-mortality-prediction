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

    x = load_data("train_x.csv")
    y = load_data("train_y.csv")

    #train_x, train_y, test_x, test_y = split_data(x, y)

    ###### Your Code Here #######
    # Add anything you want here

    ############################

    #processed_x_train = preprocess_x(train_x)
    x = preprocess_x(x)
    #processed_x_test = preprocess_x(test_x)
    processed_x_train, train_y, processed_x_test, test_y = split_data(x, y)
    ###### Your Code Here #######
    # Add anything you want here

    ### preprocessing y 
    #combine x and y based on id
    train_y = train_y.iloc[: , 1:]
    df_model = pd.merge(processed_x_train, y, left_index=True, right_index=True)

    processed_x_train = df_model.iloc[: , : df_model.shape[1]-1]
    train_y = df_model.iloc[:,df_model.shape[1]-1]


    #get the columns to be same order
    cols = list(processed_x_train.columns)
    processed_x_test = processed_x_test.reindex(columns = cols)

    #standardize
    scaler = StandardScaler()
    scaler.fit(processed_x_train)
    processed_x_train = scaler.transform(processed_x_train)

    scaler.fit(processed_x_test)
    processed_x_test = scaler.transform(processed_x_test)
    ############################

    model = Model(args)  # you can add arguments as needed
    model.fit(processed_x_train, train_y)
    x = load_data("test_x.csv")

    ###### Your Code Here #######
    # Add anything you want here

    ############################

    processed_x_test = preprocess_x(x)

    #get the columns to be same order
    cols = list(processed_x_train.columns)
    processed_x_test = processed_x_test.reindex(columns = cols)
    X_sub_copy = processed_x_test
    # Fit the scaler to the data
    scaler.fit(processed_x_test)
    # Transform the training and testing data
    processed_x_test = scaler.transform(processed_x_test)

    prediction_probs = model.predict_proba(processed_x_test)

    #### Your Code Here ####
    # Save your results
    X_sub_copy['hospitaldischargestatus'] = prediction_probs
    X_sub_copy = X_sub_copy.reset_index().rename(columns={'index': 'indexpatientunitstayid_col'})
    output = pd.concat([X_sub_copy['patientunitstayid'], X_sub_copy['hospitaldischargestatus']], axis=1)
    output["patientunitstayid"] = output["patientunitstayid"].astype(int)
    #print(output)
    output.to_csv('submission.csv', index = False)
    ########################


if __name__ == "__main__":
    main()
