# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

import numpy as np

from amal_first_kedro.pipelines.data_science.plot import plot_confusion_matrix, cv_plot, plot_auc


def train_logistic_regression(selected_features: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, X_train_final: pd.DataFrame, X_val_final: pd.DataFrame):
    """Logistic regression with selected features"""

    alpha = np.logspace(-4,4,9)
    cv_auc_score = []
    text_results = ""

    # initial Stochastic Gradient Descent model training until finding the best parameter alpha
    for i in alpha:
        clf = SGDClassifier(alpha=i, penalty='l1',class_weight = 'balanced', loss='log', random_state=28)
        clf.fit(X_train_final[selected_features], y_train)
        sig_clf = CalibratedClassifierCV(clf, method='sigmoid')
        sig_clf.fit(X_train_final[selected_features], y_train)
        y_pred_prob = sig_clf.predict_proba(X_val_final[selected_features])[:,1]
        cv_auc_score.append(roc_auc_score(y_val,y_pred_prob))

        text_results += 'For alpha {0}, cross validation AUC score {1}'.format(i,roc_auc_score(y_val,y_pred_prob)) + "\n"
        print('For alpha {0}, cross validation AUC score {1}'.format(i,roc_auc_score(y_val,y_pred_prob)))
    
    text_results += 'The Optimal C value is:' + str(alpha[np.argmax(cv_auc_score)])
    print('The Optimal C value is:', alpha[np.argmax(cv_auc_score)])    

    # trainig the Stochastic Gradient Descent Classifier using the best alpha
    best_alpha = alpha[np.argmax(cv_auc_score)]
    logreg = SGDClassifier(alpha = best_alpha, class_weight = 'balanced', penalty = 'l1', loss='log', random_state = 28)
    logreg.fit(X_train_final[selected_features], y_train)
    logreg_sig_clf = CalibratedClassifierCV(logreg, method='sigmoid')
    logreg_sig_clf.fit(X_train_final[selected_features], y_train)

    cv_auc_score = cv_plot(alpha, cv_auc_score)

    return logreg, logreg_sig_clf, best_alpha, cv_auc_score, text_results

def predict(logreg_sig_clf, logreg, best_alpha, selected_features: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame, X_train_final: pd.DataFrame, X_val_final: pd.DataFrame, X_test_final: pd.DataFrame):
    text_results = ""
    y_pred_prob = logreg_sig_clf.predict_proba(X_train_final[selected_features])[:,1]

    text_results += "\n \n \n \nLogistic Regression model results after finding the best alpha \n"
    text_results += 'For best alpha {0}, The Train AUC score is {1}'.format(best_alpha, roc_auc_score(y_train,y_pred_prob) ) + "\n"
    
    y_pred_prob = logreg_sig_clf.predict_proba(X_val_final[selected_features])[:,1]
    text_results += 'For best alpha {0}, The Cross validated AUC score is {1}'.format(best_alpha, roc_auc_score(y_val,y_pred_prob) ) + "\n"

    y_pred_prob = logreg_sig_clf.predict_proba(X_test_final[selected_features])[:,1]
    text_results += 'For best alpha {0}, The Test AUC score is {1}'.format(best_alpha, roc_auc_score(y_test,y_pred_prob) ) + "\n"

    y_pred = logreg.predict(X_test_final[selected_features])
    text_results += 'The test AUC score is :' + str( roc_auc_score(y_test,y_pred_prob)) + "\n"

    text_results += 'The percentage of misclassified points {:05.2f}% :'.format((1-accuracy_score(y_test, y_pred))*100) + "\n"

    return y_pred, text_results, y_pred_prob

def classification_report(y_test, y_pred, y_pred_prob):

    confusion_matrix_fig = plot_confusion_matrix(y_test, y_pred)
    auc_curve = plot_auc(y_test, y_pred_prob)

    return confusion_matrix_fig, auc_curve





    
