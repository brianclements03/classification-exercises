import pandas as pd
import numpy as np
import matplotlib as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquire

def clean_titanic_data(df):
    '''
    This function will clean the data etc etc...
    '''
    
    df = df.drop_duplicates()
    cols_to_drop = ['deck', 'embarked', 'class']
    df = df.drop(columns = cols_to_drop)
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    df['baseline_prediction'] = 0
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=[True,True])
    df = pd.concat([df, dummy_df], axis = 1)
    return df

def split_titanic_data(df):
    '''
    Takes in a dataframe and returns train, validate, test sbusert dataframes
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)
    return train, validate, test

def impute_mode(train, validate, test):
    '''
    takes in train, validate, and test and uses train to identify the gbset value to replace nulls in embark_town
    imputes that value in to  all three sets and returns all three sets
    '''
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    imputer = imputer.fit(train[['embark_town']])
    train[['embark_town']] = imputer.transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test

def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test

def prep_titanic_data(df):
    '''
    Combines the clean_titanic_data, split_titanic_data, and impute_mean_age functions.
    '''
    df = clean_titanic_data(df)

    train, validate, test = split_titanic_data(df)
    
    train, validate, test = impute_mean_age(train, validate, test)

    return train, validate, test


def prep_telco(df):
    df = df.drop_duplicates()
    cols_to_drop = ['payment_type_id', 'internet_service_type_id', 'contract_type_id']
    df = df.drop(columns = cols_to_drop)
    df.total_charges = df.total_charges.replace(' ',0)
    df.total_charges = df.total_charges.astype(float)
    cols_to_dummy = df[['gender','partner','dependents','phone_service','multiple_lines',
                        'online_security','online_backup','device_protection','tech_support','streaming_tv',
                        'streaming_movies','paperless_billing','churn','contract_type',
                        'internet_service_type','payment_type']]
    dummy_df = pd.get_dummies(cols_to_dummy, dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis = 1)

    return df

def split_telco_data(df):
    '''
    Takes in a dataframe and returns train, validate, test sbusert dataframes
    '''
    telco_train, telco_test = train_test_split(df, test_size = .2, stratify=df.churn_Yes)
    telco_train, telco_validate = train_test_split(telco_train, test_size=.3, stratify=telco_train.churn_Yes)
    return telco_train, telco_validate, telco_test

