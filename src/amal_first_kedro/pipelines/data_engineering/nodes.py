import pandas as pd
import numpy as np

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