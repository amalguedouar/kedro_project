
from kedro.pipeline import Pipeline, node

from .nodes import limit_size, add_income_gt_credit_flag, add_credit_income_percent, add_annuity_income_percent, add_credit_term, add_days_deployed_percent, create_application_bureau_data, feature_engineering_bureau_application, create_application_bureau_prev

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=limit_size,
                inputs=["application_train_data","params:limit_size"],
                outputs="application_train_data_limited",
                name="limit_size",
            ),
            node(
                func=add_income_gt_credit_flag,
                inputs=["application_train_data_limited"],
                outputs="application_train_data_income_flag",
                name="add_column_1",
            ),
            node(
                func=add_credit_income_percent,
                inputs=["application_train_data_income_flag"],
                outputs="application_train_data_credit_income_percent",
                name="add_column_2",
            ),
            node(
                func=add_annuity_income_percent,
                inputs=["application_train_data_credit_income_percent"],
                outputs="application_train_data_annuity_income_percent",
                name="add_column_3",
            ),
                node(
                func=add_credit_term,
                inputs=["application_train_data_annuity_income_percent"],
                outputs="application_train_data_credit_term",
                name="add_column_4",
            ),
                node(
                func=add_days_deployed_percent,
                inputs=["application_train_data_credit_term"],
                outputs="application_train_data_days_deployed_percent",
                name="add_column_5",
            ),
                node(
                func=create_application_bureau_data,
                inputs=["bureau_data", "application_train_data_days_deployed_percent"],
                outputs="application_bureau_data",
                name="create_application_bureau_data",
            ),
                node(
                func=feature_engineering_bureau_application,
                inputs=["bureau_data", "application_bureau_data"],
                outputs="application_bureau_fe",
                name="feature_engineering_bureau_application",
            ),
                node(
                func=create_application_bureau_prev,
                inputs=["previous_application", "application_bureau_fe"],
                outputs="application_bureau_prev",
                name="create_application_bureau_prev",
            ),
        ]
    )