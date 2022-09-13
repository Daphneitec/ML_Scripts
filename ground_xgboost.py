##import xgboost as xgb
from xgboost import XGBClassifier as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
import matplotlib.pylab as pl
import pandas as pd
import sqlalchemy
import numpy as np
import sklearn.metrics as metrics
import pyodbc
import pymssql
import warnings

warnings.filterwarnings("ignore")
conn = pymssql.connect(server="MatrixSQLSIT", user="SCI_ML", password="Matr1xSCIML",
                       database="SCI_ML_SIT")
df = pd.read_sql(
    'select  ROUTE_TYPE, LANE_NAME, O_Entity_Type, D_Entity_Type, MODE_SERVICE_LEVEL_TYPE, O_Zone, D_Zone, (Transit_Days -1) as Transit_Days from dbo.v_GTC_Transitdays_Zone_Training where O_Zone is not null and D_Zone is not null',
    conn)

t = df.shape[0]
s = t - 2500  ##use 2500 for test

cols_to_drop = ['Transit_Days']
cols_to_encode = ['ROUTE_TYPE', 'LANE_NAME', 'O_Entity_Type', 'D_Entity_Type', 'MODE_SERVICE_LEVEL_TYPE', 'O_Zone', 'D_Zone']

edf = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

##Print result
df_test = df.iloc[s:t]

y = df.loc[:,'Transit_Days']
X = edf.drop(cols_to_drop, axis='columns')

X_train = X.iloc[0:s]
X_test = X.iloc[s:t]
y_train = y.iloc[0:s]
y_test = y.iloc[s:t]


##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##le = LabelEncoder()
##y_train = le.fit_transform(y_train)
xgbc = xgb(learning_rate=0.5,
                    n_estimators=150,
                    max_depth=6,
                    min_child_weight=0,
                    gamma=0,
                    reg_lambda=1,
                    subsample=1,
                    colsample_bytree=0.75,
                    scale_pos_weight=1,
                    objective='multi:softprob',
                    num_class=14,
                    random_state=42)
mcl = xgbc.fit(X_train, y_train, eval_metric='mlogloss')
pred = mcl.predict(X_test)

df_test['Predict'] = pred    ## [x+1 for x in pred]
print(df_test)
engine = sqlalchemy.create_engine("mssql+pymssql://SCI_ML:Matr1xSCIML@MatrixSQLSIT/SCI_ML_SIT")

df_test.to_sql('GTC_XGBClassifier_Transitday_Predict', engine, schema='dbo', if_exists='replace', index=False)

##print(pred)

##print(X.head())
##print(df.head())
##multi:softmax or multi:softprob

