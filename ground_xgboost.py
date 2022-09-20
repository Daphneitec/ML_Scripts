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
import pickle

warnings.filterwarnings("ignore")
conn = pymssql.connect(server="MatrixSQLSIT", user="SCI_ML", password="Matr1xSCIML",
                       database="SCI_ML_SIT")
df = pd.read_sql(
    'select  ROUTE_TYPE, LANE_NAME, O_Entity_Type, D_Entity_Type, MODE_SERVICE_LEVEL_TYPE, O_Zone, D_Zone, (Transit_Days -1) as Transit_Days from dbo.v_GTC_Transitdays_Zone_Training where O_Zone is not null and D_Zone is not null',
    conn)


t = df.shape[0]
s = t - 2500  ##use the last 2500 for test

cols_to_drop = ['Transit_Days']
cols_to_encode = ['ROUTE_TYPE', 'LANE_NAME', 'O_Entity_Type', 'D_Entity_Type', 'MODE_SERVICE_LEVEL_TYPE', 'O_Zone', 'D_Zone']

edf = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

##Print Data Columns
for col in edf.columns:
    print(col)

##Print result
df_test = df.iloc[s:t]

y = df.loc[:,'Transit_Days']
X = edf.drop(cols_to_drop, axis='columns')
print(len(edf.columns))

X_train = X.iloc[0:s]
X_test = X.iloc[s:t]
y_train = y.iloc[0:s]
y_test = y.iloc[s:t]


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
print(mcl.feature_importances_)
data = {'Feature': X_train.columns,
        'Importance': mcl.feature_importances_
        }

plotdata = pd.DataFrame(data, columns=['Feature', 'Importance'])

plotdata.plot(x='Feature', y='Importance', kind="bar")
pl.show()

##pl.bar(range(len(mcl.feature_importances_)), mcl.feature_importances_)
#pl.show()
pred = mcl.predict(X_test)
df_test['Predict'] = pred    ## [x+1 for x in pred]
print(df_test)
engine = sqlalchemy.create_engine("mssql+pymssql://SCI_ML:Matr1xSCIML@MatrixSQLSIT/SCI_ML_SIT")
df_test.to_sql('GTC_XGBClassifier_Transitday_Predict', engine, schema='dbo', if_exists='replace', index=False)

##Save Model to SQL server
model_dump = pickle.dumps(mcl)
cursor = conn.cursor()
cursor.execute("delete from [dbo].[trained_models] where model_name = 'XGBClassifier' ")
cursor.execute("insert into dbo.trained_models (model_name,model) values ( 'XGBClassifier', %s)", model_dump)
conn.commit()



### for API or store procedure call for one prediction every time
ddf = pd.read_sql(
    'select distinct ROUTE_TYPE, LANE_NAME, O_Entity_Type, D_Entity_Type, MODE_SERVICE_LEVEL_TYPE, O_Zone, D_Zone from dbo.v_GTC_Transitdays_Zone_Training where O_Zone is not null and D_Zone is not null',
    conn)
##data = ['Line Haul','Fleet Schedule','Station','Station','Default Service Level','Zone B','Zone A']
##Truckload	FTL - DYNAMIC	Station	Customer	DIRECT	Zone A	Zone H
data = ['Truckload','FTL - DYNAMIC','Station','Customer','DIRECT','Zone A','Zone H']
ddf.loc[len(ddf.index)] = data
print(ddf)
pdf = pd.get_dummies(ddf, drop_first=True)
pdf = pdf.tail(1)
print(pdf)
mypred = mcl.predict(pdf)
presult = ddf.tail(1)
presult['Predict'] = mypred
print(presult)



##print(pred)

##print(X.head())
##print(df.head())
##multi:softmax or multi:softprob


