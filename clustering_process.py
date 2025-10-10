import pandas as pd
import re
from preprocessing import compute_travel_efficiency,fetch_transport_mode
import numpy as np
import csv

PERSONS_COLUMNS = 'agegroup,sex,carlicence,mbikelicence,otherlicence,nolicence,fulltimework,parttimework,casualwork,anywork,numstops,persinc,wfhtravday,anytoll,anyvehwalk,anypaidpark,dayType'.split(',')
PER_COL_BOOL='carlicence,mbikelicence,otherlicence,nolicence,fulltimework,parttimework,casualwork,anywork,wfhtravday,anytoll,anyvehwalk,anypaidpark'.split(',')
COL_BOOL_IRREGULAR='sex,dayType'.split(',')
PER_WEIGHT="perspoststratweight"
WFH_COLUMNS=["wfhmon", "wfhtue", "wfhwed", "wfhthu","wfhfri", "wfhsat", "wfhsun"]
IDS='persid,hhid,persno'.split(',')



TRIPS_COL =['persid','triptime','travtime','cumdist','trippoststratweight']


def processed_data_clustering(using_columns=PERSONS_COLUMNS):

    df = pd.read_csv("datasets/persons.csv", usecols=IDS+PERSONS_COLUMNS+[PER_WEIGHT]+WFH_COLUMNS)
    df,irregular_bool_initial=boolean_conversion(df,using_columns)
    df['totalwfh'] = df[WFH_COLUMNS].sum(axis=1)
    df[PER_WEIGHT] = pd.to_numeric(df[PER_WEIGHT], errors='coerce')
    df,normalized_scaling = string_int_averaging(df,using_columns)
    
    wfh_dropped=[x for x in WFH_COLUMNS if x not in using_columns]
    df.drop(columns=wfh_dropped, axis=1, inplace=True)

    trips = pd.read_csv("datasets/trips.csv", usecols=TRIPS_COL)

    trips[['triptime','travtime','cumdist','trippoststratweight']] = trips[['triptime','travtime','cumdist','trippoststratweight']].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
    trips=trips.apply(lambda x : [x['persid']]+[x['trippoststratweight']*y for y in x[['triptime','travtime','cumdist']]]+[x['trippoststratweight']],axis=1,result_type='broadcast')
    
    trips["wasted_time"] = pd.to_numeric(trips["triptime"], errors='coerce') - pd.to_numeric(trips["travtime"], errors='coerce')
    
    trips["overall_trip_efficiency"] = pd.to_numeric(trips["cumdist"], errors='coerce') / pd.to_numeric(trips["travtime"], errors='coerce')
    trips['time']=trips['travtime']
    trips['distance']=trips['cumdist']





    
    # tidy dataframe
    df = df.merge(trips, on='persid', how='left')
    
    df = df.dropna()  
    df = df[['persid',PER_WEIGHT]+using_columns]
    df,mode_scaling=trip_mode_calculations(df)
    return df,irregular_bool_initial,normalized_scaling,mode_scaling

def boolean_conversion(df,using_columns):
    overlap_columns = list(set(using_columns).intersection(PER_COL_BOOL))+WFH_COLUMNS
    df[overlap_columns]=df[overlap_columns].replace({'Yes': 1, 'No': 0})
    df[overlap_columns] = df[overlap_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    overlap_columns = set(using_columns).intersection(COL_BOOL_IRREGULAR)
    irregular_bool_initial={}
    for column in overlap_columns:
        #assumes that all columns values belong solely to one of 2 distinct values
        init_value = df.head(1)[column][0]
        irregular_bool_initial[column]=init_value
        df[column]=df[column].replace({init_value: 1})
        df[column] = df[column].apply(pd.to_numeric, errors='coerce').fillna(0) 
    return df,irregular_bool_initial

FORMATING_COLUMNS=['agegroup','persinc']

def string_int_averaging(df,using_columns):
    formatable = list(set(FORMATING_COLUMNS).intersection(using_columns))
    maximums={}
    for type in formatable:
        if type=='agegroup':
            pattern_unit = r'(\d{1,3})'
            tot_pattern = r'(\d{1,3})->(\d{1,3})'
            df[type]=df[type].apply(lambda x: sub_function(x,pattern_unit,tot_pattern))
        elif type =='persinc':
            pattern=r'\$(\d{1,6})-\$(\d{1,6}) \(\$\d{1,6}-\$\d{1,6}\)'
            pattern_unit=r'(\d{1,6})'
            df[type]=df[type].apply(lambda x: sub_function(x,pattern_unit,pattern))
        max_in_type=df[type].max()
        
        maximums[type]=max_in_type

    return df,maximums
def sub_function(input,pattern_unit,split_pattern):
    input=re.sub(r',','',input)
    if re.search(split_pattern, input):
        found = re.split(split_pattern,input)
        found = [x for x in found if x.strip()]
        splitted= [float(x) for x in found[:2]]
        return sum(splitted)/(len(splitted))
    elif re.search(pattern_unit+r'+', input):
        found = re.split(pattern_unit+r'+',input)
        return float(found[1])
    else:
        # NIL AND NEGATIVE INCOME DEFAULT TO ZERO
        return 0


ALL_POSSIBLE_MODES='Vehicle Driver;Vehicle Passenger;Motorcycle;Walking;Bicycle;Taxi;Train;Tram;School Bus;Public Bus;Plane;Other;Mobility Scooter;Rideshare Service;e-Scooter;Running/jogging;Not applicable'.split(';')
MODES='mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9'.split(',')
ALLOWED_MODES='Vehicle Driver;Vehicle Passenger;Motorcycle;Walking;Bicycle;Taxi;Train;Tram;School Bus;Public Bus;Plane;Other;Mobility Scooter;Rideshare Service;e-Scooter;Running/jogging'.split(';')
def trip_mode_calculations(df):
    cols = 'persid,mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9,trippoststratweight'.split(',')
    
    trip_df=pd.read_csv("datasets/trips.csv", usecols=cols)
    trip_df[ALL_POSSIBLE_MODES]= [0] * len(ALL_POSSIBLE_MODES)
    trip_df=trip_df.apply(lambda x: adding(x),axis=1,result_type='broadcast')
    
    trip_df.drop(columns=['Not applicable','trippoststratweight']+MODES, axis=1, inplace=True)
    
    normalising_scaling=trip_df[ALLOWED_MODES].max().max()
    trip_df[ALLOWED_MODES]=trip_df[ALLOWED_MODES].apply(lambda x: x/normalising_scaling)
    df = df.merge(trip_df, on='persid', how='left')
    return df,normalising_scaling
def adding(input):
    weight = input['trippoststratweight']
    for mode in MODES:
        input[input[mode]]=input[input[mode]]+weight
    return input

def most_often_mode(df):
    
    df['most_often_mode']=df[ALLOWED_MODES].apply(lambda x: ALLOWED_MODES[np.argmax(x)],axis=1,result_type='reduce').values

    return df

def readable(df,name):
    with open(name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(df.columns)
        writer.writerows(df.values)
    f.close()