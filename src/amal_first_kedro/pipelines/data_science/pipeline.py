from kedro.pipeline import Pipeline, node

from amal_first_kedro.pipelines.data_science.feature_selection import  select_features
from amal_first_kedro.pipelines.data_science.logistic_regression import train_logistic_regression, predict, classification_report


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=select_features,
                inputs=["X_train_final","y_train"],
                outputs="selected_features",
                name="select_features",
            ),
            node(
                func=train_logistic_regression,
                inputs=["selected_features", "y_train", "y_val", "X_train_final", "X_val_final"],
                outputs=["logreg", "logreg_sig_clf", "best_alpha", "cv_auc_score", "best_alpha_results"],
                name="logistic_regression",
            ),
            node(
                func=predict,
                inputs=["logreg_sig_clf", "logreg", "best_alpha", "selected_features", "y_train", "y_val", "y_test", "X_train_final", "X_val_final", "X_test_final"],
                outputs=["y_pred", "logistic_regression_results", "y_pred_prob"],
                name="predict",
            ),
            node(
                func=classification_report,
                inputs=["y_test", "y_pred", "y_pred_prob"],
                outputs=["confusion_matrix_fig", "auc_curve"],
                name="classification_report",
            ),                       
        ]
    )



