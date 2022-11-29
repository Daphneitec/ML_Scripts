import numpy as np
import pandas as pd
import requests
import json
import xml.etree.ElementTree as ET
import pyodbc
import pandas as pd
import sqlalchemy
import pymssql
import warnings

warnings.filterwarnings("ignore")

# conn = pyodbc.connect('Driver={SQL Server};'
#                      'Server=MatrixSQLSIT;'
#                      'Database=SCI_ML_SIT;'
#                     'Trusted_Connection=yes;')

# cursor = conn.cursor()

# engine = sqlalchemy.create_engine("mssql+pymssql://SCI_ML:Matr1xSCIML@MatrixSQLSIT/SCI_ML_SIT")
# conn = engine.connect()
conn = pymssql.connect(server="MatrixSQLSIT", user="SCI_ML", password="Matr1xSCIML",
                       database="SCI_ML_SIT")  # You can lookup the port number inside SQL server.

# df = pd.read_sql_query('select  distinct House_Bill, MASTER_Airway_Bill from [dbo].[Nike_399_MAM_Days_RAW] where delivery_DATE = \'2021-12-1\' ', conn)
# df = pd.read_sql('select  distinct House_Bill, MASTER_Airway_Bill from [dbo].[Nike_399_MAM_Days_RAW]  where MASTER_Airway_Bill is not null and House_Bill is not null ', conn)
df = pd.read_sql(
    'select  distinct House_Bill, MASTER_Airway_Bill from [dbo].[Nike_399_MAM_Days_RAW]  where MASTER_Airway_Bill is not null ',
    conn)
# df = pd.read_sql('select  distinct House_Bill, MASTER_Airway_Bill from [dbo].[Nike_399_MAM_Days_RAW]  where MASTER_Airway_Bill in ( \'15777058844\' , \'11282998112\', \'20581403431\', \'23545998853\' ) '  , conn)
# df = pd.read_sql('select  distinct House_Bill, MASTER_Airway_Bill from [dbo].[Nike_399_MAM_Days_RAW]  where MASTER_Airway_Bill in (  \'15777058844\' , \'23545998853\' ) '  , conn)

myrow = 0
for ind in df.index:
    url = "https://transportleg-rest.kus.logistics.corp/v2/transportlegs?masterNumber=" + df['MASTER_Airway_Bill'][ind]
    response = requests.get(url, verify=False)
    mydata = []

    #    print(df['MASTER_Airway_Bill'][ind])
    if response.status_code == 204:
        mymaster = df['MASTER_Airway_Bill'][ind]
        mymaster = mymaster[0:3] + '-' + mymaster[3:]
        url = "https://transportleg-rest.kus.logistics.corp/v2/transportlegs?masterNumber=" + mymaster
        #               print(url)
        response = requests.get(url, verify=False)

    #    print('resp_code',response.status_code)
    if response.status_code == 204:
        #        print('break here---------------------')
        continue
    else:
        request = response.json()
        #        print("---------------------------------------------------------------------------------")
        items = request['transport_leg']
        item = items[0]
        #        if not ("house_number" in item) : continue

        #        if ( df['House_Bill'][ind] in item['house_number']  )  :

        myxml = item['transport_leg_data'].replace(
            "version=\"2.5\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns=\"http://cevalogistics.com/OFS/Schemas/BookingUpdate\"",
            "")

        myroot0 = ET.fromstring(myxml)
        #               for elem in myroot0.iter() :
        #               print(elem)
        ##[elem.tag for elem in root.iter()] find all iter elements
        #                for child in myroot0 :
        #                        child = myroot0[0]
        child = myroot0
        #               print(child)
        #               if child.tag == 'MasterBooking' :
        route1 = ""
        for routeto in child.iter('Route1To'):
            route1 = routeto.text.lstrip().rstrip()
        # print('Route1To:',routeto.text)
        route2 = ""
        for routeto in child.iter('Route2To'):
            route2 = routeto.text.lstrip().rstrip()
        # print('Route2To:', routeto.text)
        origin = ""
        for routeto in child.iter('From'):
            origin = routeto.text.lstrip().rstrip()
        # print('From:',routeto.text)
        dest = ""
        for routeto in child.iter('To'):
            dest = routeto.text.lstrip().rstrip()
        # print('To:',routeto.text)
        carrier_code = ""
        for routeto in child.iter('CarrierCode'):
            carrier_code = routeto.text.lstrip().rstrip()
        carrier_name = ""
        for routeto in child.iter('CarrierName'):
            carrier_name = routeto.text.lstrip().rstrip()
        if route2 == dest:
            myroute = origin + '-' + route1 + '-' + route2
        else:
            myroute = origin + '-' + route1 + '-' + route2 + '-' + dest
        if "carrier" in item:
            myairline = carrier_code
        else:
            myairline = ""

        mydict = {'Master_Bill_No': df['MASTER_Airway_Bill'][ind]
            , 'House_Bill_No': df['House_Bill'][ind]
            , 'Air_Path': myroute
            , 'Airline_Code': myairline
            , 'Airline_Name': carrier_name}

        if myrow == 0:
            mydf = pd.DataFrame([mydict])
        else:
            new_row = pd.Series(mydict)
            mydf = mydf.append(new_row, ignore_index=True)
        myrow = myrow + 1

    #        else:
#           continue

print(mydf)

engine = sqlalchemy.create_engine("mssql+pymssql://SCI_ML:Matr1xSCIML@MatrixSQLSIT/SCI_ML_SIT")

mydf.to_sql('Master_Entities_Airpath', engine, schema='dbo', if_exists='replace', index=False)