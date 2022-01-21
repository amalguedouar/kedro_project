import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def limit_size(application_train_data: pd.DataFrame, size: int) -> pd.DataFrame:
    """Preprocesses the data for yellow tripdata.

    Args:
        the application_train_data and the limit size
    Returns:
        the limited data frame.
    """
    if size is None:
        print("No limit size provided. Full data will be returned")
        return application_train_data

    limited_dataframe = application_train_data.head(size)
    return limited_dataframe

def add_income_gt_credit_flag(application_train_data: pd.DataFrame) -> pd.DataFrame:
    """Flag to represent when Total income is greater than Credit

    Args: 
        the application_train_data.
    Returns:
        the application_train_data with new credit flag column.
    """
    # Create an error flag column
    application_train_data['DAYS_EMPLOYED_ERROR'] = application_train_data["DAYS_EMPLOYED"] == 365243
    # Replace the error values with nan
    application_train_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

    application_train_data['INCOME_GT_CREDIT_FLAG'] = application_train_data['AMT_INCOME_TOTAL'] > application_train_data['AMT_CREDIT']
    return application_train_data

def add_credit_income_percent(application_train_data: pd.DataFrame) -> pd.DataFrame:
    """Column to represent Credit Income Percent.
    Args: 
        the application_train_data.
    Returns:
        the application_train_data with new Credit Income Percent column.
    """

    application_train_data['CREDIT_INCOME_PERCENT'] = application_train_data['AMT_CREDIT'] / application_train_data['AMT_INCOME_TOTAL']
    return application_train_data


def add_annuity_income_percent(application_train_data: pd.DataFrame) -> pd.DataFrame:
    """Column to represent Annuity Income percent.
    Args: 
        the application_train_data.
    Returns:
        the application_train_data with new ANNUITY_INCOME_PERCENT column.
    """
        
    application_train_data['ANNUITY_INCOME_PERCENT'] = application_train_data['AMT_ANNUITY'] / application_train_data['AMT_INCOME_TOTAL']
    return application_train_data


def add_credit_term(application_train_data: pd.DataFrame) -> pd.DataFrame:
    """Column to represent credit term.
    Args: 
        the application_train_data.
    Returns:
        the application_train_data with new CREDIT_TERM column.
    """
        
    application_train_data['CREDIT_TERM'] = application_train_data['AMT_CREDIT'] / application_train_data['AMT_ANNUITY']
    return application_train_data

def add_days_deployed_percent(application_train_data: pd.DataFrame) -> pd.DataFrame:
    """Column to represent Days Employed percent in his life
.
    Args: 
        the application_train_data.
    Returns:
        the application_train_data with new DAYS_EMPLOYED_PERCENT column.
    """

    application_train_data['DAYS_EMPLOYED_PERCENT'] = application_train_data['DAYS_EMPLOYED'] / application_train_data['DAYS_BIRTH']
    return application_train_data



# not used yet
def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table


def create_application_bureau_data(
    bureau: pd.DataFrame, application: pd.DataFrame) -> pd.DataFrame:
    """Joining Bureau data to Application data.
    Args:
        bureau: Input bureau data.
        application: Preprocessed data for application.
    Returns:
        merged preprocessed application table with input bureau table.

    """

    # Combining numerical features
    grp = bureau.drop(['SK_ID_BUREAU'], axis = 1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]
    application_bureau = application.merge(grp, on='SK_ID_CURR', how='left')
    application_bureau.update(application_bureau[grp.columns].fillna(0))

    # Combining categorical features
    bureau_categorical = pd.get_dummies(bureau.select_dtypes('object'))
    bureau_categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
    grp = bureau_categorical.groupby(by = ['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]
    application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
    application_bureau.update(application_bureau[grp.columns].fillna(0))

    # Shape of application and bureau data combined
    print('The shape application and bureau data combined:',application_bureau.shape)

    return application_bureau

def feature_engineering_bureau_application(
    bureau: pd.DataFrame, application_bureau: pd.DataFrame) -> pd.DataFrame:
    """Feature Engineering of Bureau Data.
    Args:
        bureau: Input bureau data.
        application_bureau: merged preprocessed application table with input bureau table.
    Returns:
        bureau application table after feature engineering.
    """

    # Number of past loans per customer
    grp = bureau.groupby(by = ['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index().rename(columns = {'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'})
    application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
    application_bureau['BUREAU_LOAN_COUNT'] = application_bureau['BUREAU_LOAN_COUNT'].fillna(0)

    # Number of types of past loans per customer 
    grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
    application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
    application_bureau['BUREAU_LOAN_TYPES'] = application_bureau['BUREAU_LOAN_TYPES'].fillna(0)

    # Debt over credit ratio 
    bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(0)
    bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
    grp1 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM': 'TOTAL_CREDIT_SUM'})
    grp2 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CREDIT_SUM_DEBT'})
    grp1['DEBT_CREDIT_RATIO'] = grp2['TOTAL_CREDIT_SUM_DEBT']/grp1['TOTAL_CREDIT_SUM']
    del grp1['TOTAL_CREDIT_SUM']
    application_bureau = application_bureau.merge(grp1, on='SK_ID_CURR', how='left')
    application_bureau['DEBT_CREDIT_RATIO'] = application_bureau['DEBT_CREDIT_RATIO'].fillna(0)
    application_bureau['DEBT_CREDIT_RATIO'] = application_bureau['DEBT_CREDIT_RATIO'].replace([np.inf, -np.inf], 0)
    application_bureau['DEBT_CREDIT_RATIO'] = pd.to_numeric(application_bureau['DEBT_CREDIT_RATIO'], downcast='float')
    
    # Overdue over debt ratio
    bureau['AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0)
    bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
    grp1 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
    grp2 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CUSTOMER_DEBT'})
    grp1['OVERDUE_DEBT_RATIO'] = grp1['TOTAL_CUSTOMER_OVERDUE']/grp2['TOTAL_CUSTOMER_DEBT']
    del grp1['TOTAL_CUSTOMER_OVERDUE']
    application_bureau = application_bureau.merge(grp1, on='SK_ID_CURR', how='left')
    application_bureau['OVERDUE_DEBT_RATIO'] = application_bureau['OVERDUE_DEBT_RATIO'].fillna(0)
    application_bureau['OVERDUE_DEBT_RATIO'] = application_bureau['OVERDUE_DEBT_RATIO'].replace([np.inf, -np.inf], 0)
    application_bureau['OVERDUE_DEBT_RATIO'] = pd.to_numeric(application_bureau['OVERDUE_DEBT_RATIO'], downcast='float')

    return application_bureau

def create_application_bureau_prev(
    previous_application: pd.DataFrame, application_bureau_fe: pd.DataFrame) -> pd.DataFrame:
    """Joining Previous Application data to Application Bureau data.
    Args:
        previous_application: Input previous application data.
        application_bureau_fe: bureau application table after feature engineering.
    Returns:
        bureau application table merged with previous application data.
    """

    # Number of previous applications per customer
    grp = previous_application[['SK_ID_CURR','SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'PREV_APP_COUNT'})
    application_bureau_prev = application_bureau_fe.merge(grp, on =['SK_ID_CURR'], how = 'left')
    application_bureau_prev['PREV_APP_COUNT'] = application_bureau_prev['PREV_APP_COUNT'].fillna(0)

    # Combining numerical features
    grp = previous_application.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    prev_columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
    application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

    # Combining categorical features
    prev_categorical = pd.get_dummies(previous_application.select_dtypes('object'))
    prev_categorical['SK_ID_CURR'] = previous_application['SK_ID_CURR']
    prev_categorical.head()
    grp = prev_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]
    application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
    application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

    return application_bureau_prev


def create_application_bureau_prev_cash(
    pos_cash: pd.DataFrame, application_bureau_prev: pd.DataFrame) -> pd.DataFrame:
    """Joining POS CASH balance data to application bureau prev data.
    Args:
        pos_cash: Input pos cash balance data.
        application_bureau_prev: bureau application table merged with previous application data.
    Returns:
        application_bureau_prev merged with  POS CASH balance data.
    """

    # Combining numerical features
    grp = pos_cash.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    prev_columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
    application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

    # Combining categorical features
    pos_cash_categorical = pd.get_dummies(pos_cash.select_dtypes('object'))
    pos_cash_categorical['SK_ID_CURR'] = pos_cash['SK_ID_CURR']
    grp = pos_cash_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]
    application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
    application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

    return application_bureau_prev

def create_application_bureau_payments(insta_payments: pd.DataFrame, create_application_bureau_prev_cash: pd.DataFrame) -> pd.DataFrame:
    """Joining Installments Payments data to application_bureau_prev_data.

    Args:
        insta_payments: Input insta payments data.
        application_bureau_prev: bureau application table merged with previous application data.
    Returns:
        create_application_bureau_prev_cash merged with insta payments data.
    """

    # Combining numerical features and there are no categorical features in this dataset
    grp = insta_payments.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    prev_columns = ['INSTA_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    application_bureau_prev = create_application_bureau_prev_cash.merge(grp, on =['SK_ID_CURR'], how = 'left')
    application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

    return application_bureau_prev

def create_final_fe_application(credit_card_balance: pd.DataFrame, application_bureau_payments: pd.DataFrame) -> pd.DataFrame:
    """Joining Credit card balance data to application_bureau_prev data.

    Args:
        credit_card_balance: Input credit card balance data.
        application_bureau_payments: application table merged with application_bureau_payments.
    Returns:
        final feature engineering application data.
    """

    # Combining numerical features
    grp = credit_card_balance.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    prev_columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    application_bureau_prev = application_bureau_payments.merge(grp, on =['SK_ID_CURR'], how = 'left')
    application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

    # Combining categorical features
    credit_categorical = pd.get_dummies(credit_card_balance.select_dtypes('object'))
    credit_categorical['SK_ID_CURR'] = credit_card_balance['SK_ID_CURR']
    grp = credit_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]
    application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
    application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

    return application_bureau_prev


def divide_train_test_data(application_bureau_prev: pd.DataFrame, t_size:int):
    """Dividing final data into train, valid and test datasets."""
    y = application_bureau_prev.pop('TARGET').values
    X_train, X_temp, y_train, y_temp = train_test_split(application_bureau_prev.drop(['SK_ID_CURR'],axis=1), y, stratify = y, test_size=t_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify = y_temp, test_size=0.5, random_state=42)
    print('Shape of X_train:',X_train.shape)
    print('Shape of X_val:',X_val.shape)
    print('Shape of X_test:',X_test.shape)

        # Seperation of columns into numeric and categorical columns
    types = np.array([dt for dt in X_train.dtypes])
    all_columns = X_train.columns.values
    is_num = types != 'object'
    num_cols = all_columns[is_num]
    cat_cols = all_columns[~is_num]

    # Featurization of numeric data
    imputer_num = SimpleImputer(strategy='median')
    X_train_num = imputer_num.fit_transform(X_train[num_cols])
    X_val_num = imputer_num.transform(X_val[num_cols])
    X_test_num = imputer_num.transform(X_test[num_cols])
    scaler_num = StandardScaler()
    X_train_num1 = scaler_num.fit_transform(X_train_num)
    X_val_num1 = scaler_num.transform(X_val_num)
    X_test_num1 = scaler_num.transform(X_test_num)
    X_train_num_final = pd.DataFrame(X_train_num1, columns=num_cols)
    X_val_num_final = pd.DataFrame(X_val_num1, columns=num_cols)
    X_test_num_final = pd.DataFrame(X_test_num1, columns=num_cols)

    # Featurization of categorical data
    imputer_cat = SimpleImputer(strategy='constant', fill_value='MISSING')
    X_train_cat = imputer_cat.fit_transform(X_train[cat_cols])
    X_val_cat = imputer_cat.transform(X_val[cat_cols])
    X_test_cat = imputer_cat.transform(X_test[cat_cols])
    X_train_cat1= pd.DataFrame(X_train_cat, columns=cat_cols)
    X_val_cat1= pd.DataFrame(X_val_cat, columns=cat_cols)
    X_test_cat1= pd.DataFrame(X_test_cat, columns=cat_cols)
    ohe = OneHotEncoder(sparse=False,handle_unknown='ignore')
    X_train_cat2 = ohe.fit_transform(X_train_cat1)
    X_val_cat2 = ohe.transform(X_val_cat1)
    X_test_cat2 = ohe.transform(X_test_cat1)
    cat_cols_ohe = list(ohe.get_feature_names(input_features=cat_cols))
    X_train_cat_final = pd.DataFrame(X_train_cat2, columns = cat_cols_ohe)
    X_val_cat_final = pd.DataFrame(X_val_cat2, columns = cat_cols_ohe)
    X_test_cat_final = pd.DataFrame(X_test_cat2, columns = cat_cols_ohe)

    # Final complete data
    X_train_final = pd.concat([X_train_num_final,X_train_cat_final], axis = 1)
    X_val_final = pd.concat([X_val_num_final,X_val_cat_final], axis = 1)
    X_test_final = pd.concat([X_test_num_final,X_test_cat_final], axis = 1)
    print(X_train_final.shape)
    print(X_val_final.shape)
    print(X_test_final.shape)

    y = pd.DataFrame(y)
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)

    return X_train_final, X_val_final, X_test_final, y, y_train, y_val, y_test



