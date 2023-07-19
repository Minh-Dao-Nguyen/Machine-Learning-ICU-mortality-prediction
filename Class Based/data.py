import pandas as pd
import numpy as np
import os
import itertools
from sklearn.model_selection import train_test_split


def load_data(x_path):
    # Your code here
    df = pd.read_csv(x_path, index_col=0)
    return df


def split_data(x, y, split=0.8):
    # Your code here
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
    return train_x, train_y, test_x, test_y


def preprocess_x(df):
    ##for some columns specifically, remove row if data is empty
  #nursingchartvalue
  for index, val in df['nursingchartvalue'].iteritems():
    if pd.isna(val) and not(pd.isna(df.loc[index]['nursingchartcelltypevalname'])):
      #print(df.loc[index]['patientunitstayid'])
      df.drop(index, inplace = True)
  #labresult
  for index, val in df['labresult'].iteritems():
    if not(pd.isna(df.loc[index]['labname'])) and pd.isna(val):
      #print(df.loc[index]['patientunitstayid'])
      df.drop(index, inplace = True)

  # row is id number, columns are measuremnet, each cell holds a list
  df = df.groupby('patientunitstayid').agg(lambda x: list(x))

  #clean out nan value
  for index, row in df.iterrows():
      for column in df.columns:
          row[column] = [x for x in row[column] if str(x) != 'nan']
  
  #make columns for unique measurement ( labname, nurshing chartcelltypevalname, cellable)
  #labname
  lab_lst = []
  for val in df['labname']:
    for x in val:
      lab_lst.append(x)
  labname_vals = np.unique(lab_lst)
  #nursingchartcelltypevalname
  nurs_lst = []
  for val in df['nursingchartcelltypevalname']:
    for x in val:
      nurs_lst.append(x)
  nursing_vals = np.unique(nurs_lst)
  #celllabel
  cell_lst = []
  for val in df['celllabel']:
    for x in val:
      cell_lst.append(x)
  celllabel_vals = np.unique(cell_lst)

  #make columns for unique measurement
  #labname
  for i in labname_vals:
    df[i] = ''
    df[i] = df[i].apply(list)
  #nursingchartcelltypevalname
  for i in nursing_vals:
    df[i] = ''
    df[i] = df[i].apply(list)
  #celllabel
  for i in celllabel_vals:
    df[i] = ''
    df[i] = df[i].apply(list)

  #clean data
  df['gender'] = df['gender'].apply(fill_na_gender)
  df['age'] = df['age'].apply(fill_na_age)
  df['ethnicity'] = df['ethnicity'].apply(fill_na_ethnicity)

  # do flatten lists on admissionweight, admissionheight, age, gender, ethnicity, unitvisitnumber,patientunitstayid_dupe
  df["admissionheight"] = df["admissionheight"].apply(reduce_list)
  df["admissionweight"] = df["admissionweight"].apply(reduce_list)

  df["age"] = df["age"].apply(reduce_list)
  df["gender"] = df["gender"].apply(reduce_list)
  df["ethnicity"] = df["ethnicity"].apply(reduce_list)
  df["unitvisitnumber"] = df["unitvisitnumber"].apply(reduce_list)

  ########Group Data to new Columns #########
  for index, row in df.iterrows():
      #labname
      if(len(row['labname']) != 0 ):
        for i in range(len(row['labname'])):
          row[row['labname'][i]].append(row['labresult'][i])
      #nursingchartcelltypevalname
      if(len(row['nursingchartcelltypevalname']) != 0 ):
        #add data
        for i in range(len(row['nursingchartcelltypevalname'])):
          row[row['nursingchartcelltypevalname'][i]].append(row['nursingchartvalue'][i])
      #celllabel
      if(len(row['celllabel']) != 0 ):
        #add data
        for i in range(len(row['celllabel'])):
          row[row['celllabel'][i]].append(row['cellattributevalue'][i])

  
  #drop columns: cellattributevalue, celllabel, labmeasurenamesystem, labname, nursingchartcelltypevalname, nusrsingchartvalue, offset
  df.drop(columns = ['cellattributevalue', 'celllabel', 'labmeasurenamesystem', 'labname', 'nursingchartcelltypevalname', 'nursingchartvalue', 'offset', 'labresult'], inplace = True)  

  #clean capillary refill
  df['Capillary Refill'] = df['Capillary Refill'].apply(fill_na_Capillary)

  #clean the data ( get average measurements and clean age)
  df['glucose'] = df['glucose'].apply(avg_val)
  df['pH'] = df['pH'].apply(avg_val)
  df['GCS Total'] = df['GCS Total'].apply(avg_val)
  df['Heart Rate'] = df['Heart Rate'].apply(avg_val)
  df['Invasive BP Diastolic'] = df['Invasive BP Diastolic'].apply(avg_val)
  df['Invasive BP Mean'] = df['Invasive BP Mean'].apply(avg_val)
  df['Invasive BP Systolic'] = df['Invasive BP Systolic'].apply(avg_val)
  df['Non-Invasive BP Diastolic'] = df['Non-Invasive BP Diastolic'].apply(avg_val)
  df['Non-Invasive BP Mean'] = df['Non-Invasive BP Mean'].apply(avg_val)
  df['Non-Invasive BP Systolic'] = df['Non-Invasive BP Systolic'].apply(avg_val)
  df['O2 Saturation'] = df['O2 Saturation'].apply(avg_val)
  df['Respiratory Rate'] = df['Respiratory Rate'].apply(avg_val)
  df['Capillary Refill'] = df['Capillary Refill'].apply(avg_val)

  df['age'] = df['age'].apply(change_age)

  #change non num measurement using one hot encode
  gender_cols = pd.get_dummies(df['gender'], prefix='gender')
  ethicity_cols = pd.get_dummies(df['ethnicity'], prefix='ethnicity')
  capillary_cols = pd.get_dummies(df['Capillary Refill'], prefix='Capillary Refill')
  # add back to orignal df
  df = pd.concat([df, gender_cols, ethicity_cols, capillary_cols], axis=1)
  #drop old columns: ethnicity, gender, Capillary Refill
  df.drop(columns = ['ethnicity', 'gender','Capillary Refill'], inplace = True)

  #check if some cols exist then proceed
  if 'ethnicity_Native American' not in df.columns:
    df['ethnicity_Native American'] = 0.0
  if 'ethnicity_Other/Unknown' not in df.columns:
    df['ethnicity_Other/Unknown'] = 0.0
  if 'Capillary Refill_< 2 seconds' not in df.columns:
    df['Capillary Refill_< 2 seconds'] = 0.0
  if 'Capillary Refill_< 2 seconds' not in df.columns:
    df['Capillary Refill_> 2 seconds'] = 0.0
  if 'Capillary Refill_feet' not in df.columns:
    df['Capillary Refill_feet'] = 0.0
  if 'Capillary Refill_hands' not in df.columns:
    df['Capillary Refill_hands'] = 0.0
  if 'Capillary Refill_normal' not in df.columns:
    df['Capillary Refill_normal'] = 0.0
  

  #convert columns with str_ to str for model to read
  df.columns = df.columns.astype(str) 

  df = df.fillna(0.0)

  #Clean GCS
  df['GCS Total'] = df['GCS Total'].apply(fix_GCS)

  #change to float
  # convert the data in 'col1' from uint8 to float
  df['gender_Female'] = df['gender_Female'].astype('float')
  df['gender_Male'] = df['gender_Male'].astype('float')
  df['ethnicity_African American'] = df['ethnicity_African American'].astype('float')
  df['ethnicity_Asian'] = df['ethnicity_Asian'].astype('float')
  df['ethnicity_Caucasian'] = df['ethnicity_Caucasian'].astype('float')
  df['ethnicity_Hispanic'] = df['ethnicity_Hispanic'].astype('float')
  df['ethnicity_Native American'] = df['ethnicity_Native American'].astype('float')
  df['ethnicity_Other/Unknown'] = df['ethnicity_Other/Unknown'].astype('float')
  df['Capillary Refill_< 2 seconds'] = df['Capillary Refill_< 2 seconds'].astype('float')
  df['Capillary Refill_> 2 seconds'] = df['Capillary Refill_> 2 seconds'].astype('float')
  df['Capillary Refill_feet'] = df['Capillary Refill_feet'].astype('float')
  df['Capillary Refill_hands'] = df['Capillary Refill_hands'].astype('float')
  df['Capillary Refill_normal'] = df['Capillary Refill_normal'].astype('float')
  df['age'] = df['age'].astype('float')
  df['GCS Total'] = df['GCS Total'].astype('float')

  #don't be racist
  df.drop(columns = ['ethnicity_African American', 'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic', 'ethnicity_Native American', 'ethnicity_Other/Unknown'], inplace = True) 

  df.drop(columns = ['gender_Female','gender_Male'], inplace=True)

  data = df

  return data