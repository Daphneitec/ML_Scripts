import pandas as pd
from sklearn.model_selection import train_test_split
###import tensorflow as tf
import autokeras as ak
from sklearn.metrics import classification_report
import pymssql
import warnings
#import pickle
import dill as pickle
from sqlalchemy import create_engine
import urllib
import pyodbc

#mydf = pd.read_csv('GTC_Zone_Training.csv')

warnings.filterwarnings("ignore")
conn = pymssql.connect(server="localhost", user="sa", password="sqldemo123", database="ML_POC")
mydf = pd.read_sql(
    'select  TRIP_ID, ROUTE_TYPE, LANE_NAME, O_Entity_Type, D_Entity_Type, MODE_SERVICE_LEVEL_TYPE, O_Zone, D_Zone, (Transit_Days -1) as Transit_Days '
    'from dbo.v_GTC_Zone_Transit_Days_Training where O_Zone is not null and D_Zone is not null',
    conn)
#print(mydf.shape[0])
mydf.dropna()
#print(mydf.head().to_string())
print('shape[0]=', mydf.shape[0])
cols_to_drop = ['Transit_Days']
cols_to_encode = ['ROUTE_TYPE', 'LANE_NAME', 'O_Entity_Type', 'D_Entity_Type', 'MODE_SERVICE_LEVEL_TYPE', 'O_Zone', 'D_Zone']
##encoding the categorical variables
X = pd.get_dummies(mydf, columns=cols_to_encode, drop_first=True)
X = X.drop('Transit_Days', axis=1)
y = mydf.iloc[:, -1]

#print(X.shape, y.shape)
X['TRIP_ID'] = mydf['TRIP_ID']
#X = X.drop('Transit_Days', axis=0)
print('X df: ', X)
#print(y)

train_X, Val_X, train_y, Val_y = train_test_split(X,y, test_size= 0.25, random_state=14)
train_X_ = train_X.drop('TRIP_ID', axis=1)
Val_X_ = Val_X.drop('TRIP_ID', axis=1)
print('train_X_:', train_X_.shape[0], 'Val_X_', Val_X_.shape[0])

clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)
clf.fit(train_X_, train_y, epochs=10)
predict_Val_y = clf.predict(Val_X_).astype(int)

Val_X['Predict'] = predict_Val_y
print(Val_X)
myResult = Val_X[['TRIP_ID', 'Predict']]
myResult = pd.merge(myResult, mydf, on='TRIP_ID', how='inner')


#ODBC Write Table
params = urllib.parse.quote_plus(r'DRIVER={SQL Server Native Client 11.0};SERVER=localhost;DATABASE=ML_POC;UID=sa;PWD=sqldemo123')
#params = urllib.parse.quote_plus(r'DRIVER={SQL Server Native Client 11.0};SERVER=localhost;DATABASE=ML_POC; Trusted_Connection=yes')

conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
#create an engine to bulk insert the data
engine = create_engine(conn_str,fast_executemany=True)
myResult.to_sql(name='GTC_Zone_Transit_Days_Training_Predict', schema='dbo', con=engine, if_exists='replace', index=False
                , chunksize=1000)
print(myResult)

model = clf.export_model()
model.summary()
#Model Dump to a file
Pkl_Filename = "GTC_Transit_Days.pkl"
with open(Pkl_Filename, 'wb') as file:
    pickle.dump(clf, file)
#Model Dump to a Database Server


model_dump = pickle.dumps(clf)
cursor = conn.cursor()
cursor.execute("delete from [dbo].[trained_models] where model_name = 'AutoKera_Classifier' ")
cursor.execute("insert into dbo.trained_models (model_name,model) values ( 'AutoKera_Classifier', %s)", model_dump)
conn.commit()



