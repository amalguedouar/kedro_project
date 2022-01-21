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
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle
import re

"""
This is a boilerplate pipeline 'modeling_pipeline'
generated using Kedro 0.17.5
"""
def select_features(X_train_final: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
    """
    """

    # convert dataframe to np array
    y_train.to_numpy()

    # Selection of features
    model_sk = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=7, learning_rate=0.01, n_estimators= 2000, 
                    class_weight='balanced', subsample=0.9, colsample_bytree= 0.8, n_jobs=-1)
    train_features, valid_features, train_y, valid_y = train_test_split(X_train_final, y_train, test_size = 0.15, random_state = 42)
    train_features = train_features.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    valid_features = valid_features.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    model_sk.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)], eval_metric = 'auc', verbose = 200)


    # 
    feature_imp = pd.DataFrame(sorted(zip(model_sk.feature_importances_, X_train_final.columns)), columns=['Value','Feature'])
    features_df = feature_imp.sort_values(by="Value", ascending=False)
    selected_features = list(features_df[features_df['Value']>=50]['Feature'])
    # Saving the selected features into pickle file
    with open('select_features.txt','wb') as fp:
        pickle.dump(selected_features, fp)
    print('The no. of features selected:',len(selected_features))

    return selected_features
