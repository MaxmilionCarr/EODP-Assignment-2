import pandas as pd
import numpy as np

PERSONS_COLUMNS = 'persid,agegroup,sex,relationship,carlicence,mbikelicence,otherlicence,nolicence,fulltimework,parttimework,casualwork,anywork,studying,emptype,persinc,anywfh,wfhmon,wfhtue,wfhwed,wfhthu,wfhfri,wfhsat,wfhsun,homesubregion_ASGS,homeregion_ASGS'.split(',')
JOURNEY_EDUCATION_COLUMNS = 'persid,jteid,dayType,start_loc,start_stopid,start_time,start_LGA,end_loc,end_stopid,end_time,end_LGA,mainmode_desc_01,mainmode_desc_02,mainmode_desc_03,mainmode_desc_04,mainmode_desc_05,mainmode_desc_06,mainmode_desc_07,mainmode_desc_08,mainmode_desc_09,mainmode_desc_10,mainmode_desc_11,mainmode_desc_12,mainmode_desc_13,mainmode_desc_14,mainmode_desc_15,destpurp1_desc_01,destpurp1_desc_02,destpurp1_desc_03,destpurp1_desc_04,destpurp1_desc_05,destpurp1_desc_06,destpurp1_desc_07,destpurp1_desc_08,destpurp1_desc_09,destpurp1_desc_10,destpurp1_desc_11,destpurp1_desc_12,destpurp1_desc_13,destpurp1_desc_14,destpurp1_desc_15,destplace1_desc_01,destplace1_desc_02,destplace1_desc_03,destplace1_desc_04,destplace1_desc_05,destplace1_desc_06,destplace1_desc_07,destplace1_desc_08,destplace1_desc_09,destplace1_desc_10,destplace1_desc_11,destplace1_desc_12,destplace1_desc_13,destplace1_desc_14,destplace1_desc_15,startime_01,startime_02,startime_03,startime_04,startime_05,startime_06,startime_07,startime_08,startime_09,startime_10,startime_11,startime_12,startime_13,startime_14,startime_15,arrtime_01,arrtime_02,arrtime_03,arrtime_04,arrtime_05,arrtime_06,arrtime_07,arrtime_08,arrtime_09,arrtime_10,arrtime_11,arrtime_12,arrtime_13,arrtime_14,arrtime_15,vistadist_01,vistadist_02,vistadist_03,vistadist_04,vistadist_05,vistadist_06,vistadist_07,vistadist_08,vistadist_09,vistadist_10,vistadist_11,vistadist_12,vistadist_13,vistadist_14,vistadist_15,travtime_01,travtime_02,travtime_03,travtime_04,travtime_05,travtime_06,travtime_07,travtime_08,travtime_09,travtime_10,travtime_11,travtime_12,travtime_13,travtime_14,travtime_15,main_journey_mode,journey_travel_time,journey_distance,journey_elapsed_time'.split(',')
JOURNEY_WORK_COLUMNS = 'persid,jtwid,dayType,start_loc,start_stopid,start_time,start_LGA,end_loc,end_stopid,end_time,end_LGA,mainmode_desc_01,mainmode_desc_02,mainmode_desc_03,mainmode_desc_04,mainmode_desc_05,mainmode_desc_06,mainmode_desc_07,mainmode_desc_08,mainmode_desc_09,mainmode_desc_10,mainmode_desc_11,mainmode_desc_12,mainmode_desc_13,mainmode_desc_14,mainmode_desc_15,destpurp1_desc_01,destpurp1_desc_02,destpurp1_desc_03,destpurp1_desc_04,destpurp1_desc_05,destpurp1_desc_06,destpurp1_desc_07,destpurp1_desc_08,destpurp1_desc_09,destpurp1_desc_10,destpurp1_desc_11,destpurp1_desc_12,destpurp1_desc_13,destpurp1_desc_14,destpurp1_desc_15,destplace1_desc_01,destplace1_desc_02,destplace1_desc_03,destplace1_desc_04,destplace1_desc_05,destplace1_desc_06,destplace1_desc_07,destplace1_desc_08,destplace1_desc_09,destplace1_desc_10,destplace1_desc_11,destplace1_desc_12,destplace1_desc_13,destplace1_desc_14,destplace1_desc_15,startime_01,startime_02,startime_03,startime_04,startime_05,startime_06,startime_07,startime_08,startime_09,startime_10,startime_11,startime_12,startime_13,startime_14,startime_15,arrtime_01,arrtime_02,arrtime_03,arrtime_04,arrtime_05,arrtime_06,arrtime_07,arrtime_08,arrtime_09,arrtime_10,arrtime_11,arrtime_12,arrtime_13,arrtime_14,arrtime_15,vistadist_01,vistadist_02,vistadist_03,vistadist_04,vistadist_05,vistadist_06,vistadist_07,vistadist_08,vistadist_09,vistadist_10,vistadist_11,vistadist_12,vistadist_13,vistadist_14,vistadist_15,travtime_01,travtime_02,travtime_03,travtime_04,travtime_05,travtime_06,travtime_07,travtime_08,travtime_09,travtime_10,travtime_11,travtime_12,travtime_13,travtime_14,travtime_15,main_journey_mode,journey_travel_time,journey_distance,journey_elapsed_time,journey_weight,homesubregion_ASGS,homeregion_ASGS'.split(',')
TRIPS_COLUMNS = 'persid,tripno,starthour,startime,arrhour,arrtime,origplace1,origplace2,origpurp1,origpurp2,origlga,destplace1,destplace2,destpurp1,destpurp2,destlga,triptime,travtime,cumdist,linkmode,trippurp,duration,mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9,time1,time2,time3,time4,time5,time6,time7,time8,time9,dist1,dist2,dist3,dist4,dist5,dist6,dist7,dist8,dist9,trippoststratweight,trippoststratweight_GROUP_1,trippoststratweight_GROUP_2,trippoststratweight_GROUP_3,trippoststratweight_GROUP_4,trippoststratweight_GROUP_5,trippoststratweight_GROUP_6,trippoststratweight_GROUP_7,trippoststratweight_GROUP_8,trippoststratweight_GROUP_9,trippoststratweight_GROUP_10,homesubregion_ASGS,homeregion_ASGS,dayType'.split(',')


def fetch_transport_mode():
    '''
    Fetch the highest used mode of transport for each individual (persid) based on the highest time spent.
    '''
    cols = 'persid,mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9,time1,time2,time3,time4,time5,time6,time7,time8,time9'.split(',')

    # Read the dataset
    df = pd.read_csv("datasets/trips.csv", usecols=cols)

    # Define time columns
    time_columns = [f'time{i}' for i in range(1, 10)]
    mode_columns = [f'mode{i}' for i in range(1, 10)]

    # Replace 'Not applicable' with NaN in time columns
    df[time_columns] = df[time_columns].replace('Not applicable', pd.NA)

    # Convert time columns to numeric, coercing errors to NaN
    df[time_columns] = df[time_columns].apply(pd.to_numeric, errors='coerce')

    # Find the highest time value and corresponding mode for each row
    df['max_time_index'] = df[time_columns].idxmax(axis=1)
    df['highest_mode'] = df['max_time_index'].apply(lambda x: f"mode{x[-1]}" if pd.notna(x) else None)
    df['highest_mode_of_transport'] = df.apply(lambda row: row[row['highest_mode']] if pd.notna(row['highest_mode']) else None, axis=1)

    # Group by persid and find the most used mode of transport
    result = df.groupby('persid')['highest_mode_of_transport'].apply(lambda x: x.mode()[0] if not x.isna().all() else None).reset_index()
    result.columns = ['persid', 'most_used_mode']
    return result

# Uses stops and 
def compute_travel_efficiency():
    '''
    Compute travel efficiency as the average of (distance / travel time) across all trips for each individual (persid).
    '''
    df_travel = pd.read_csv("datasets/stops.csv", usecols=["persid", "travtime", "vistadist"])
    df_travel["trip_efficiency"] = pd.to_numeric(df_travel["vistadist"], errors='coerce') / pd.to_numeric(df_travel["travtime"], errors='coerce')
    df_travel["overall_trip_efficiency"] = df_travel["persid"].map(df_travel.groupby("persid")["trip_efficiency"].mean())
    df_travel = df_travel[["persid", "overall_trip_efficiency"]].drop_duplicates()
    return df_travel

def fetch_data():
    '''
    Fetch the persons dataset with an additional column for WFH frequency labels
    '''
    df = pd.read_csv("datasets/persons.csv", usecols=PERSONS_COLUMNS + ['perspoststratweight'])
    df['totalwfh'] = fetch_wfh_labels(df)
    df["perspoststratweight"] = pd.to_numeric(df["perspoststratweight"], errors='coerce')
    df.drop(columns=['anywfh', 'wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri', 'wfhsat', 'wfhsun'], axis=1, inplace=True)
    return df

def fetch_wfh_labels(df):
    """
    Return a pandas Series of WFH frequency labels
    ('Never', 'Occasional', 'Frequent', 'Always'),
    counting people not in work force as 0 WFH days.
    """
    wfh_cols = ["wfhmon", "wfhtue", "wfhwed", "wfhthu",
                "wfhfri", "wfhsat", "wfhsun"]

    # convert Yes/No to 1/0, everything else → 0
    wfh_numeric = df[wfh_cols].replace({'Yes': 1, 'No': 0})
    wfh_numeric = wfh_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)

    # total WFH days in the week
    wfhsum = wfh_numeric.sum(axis=1)

    # bin into categories
    bins   = [-1, 0, 2, 5, 7]
    labels = ['Never', 'Occasional', 'Frequent', 'Always']
    return pd.cut(wfhsum, bins=bins, labels=labels)


def categories_persons(df):
    '''
    Fetch categories for analysis of all columns
    '''
    categories = {}
    df.drop(columns=['persid'], axis=1, inplace=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            categories[col] = df[col].astype('category').cat.categories.tolist()
        elif df[col].dtype in ['int64', 'float64']:
            categories[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std()
            }
        else:
            categories[col] = 'Unsupported data type'
    return categories

ORDINAL_ENCODING = ['agegroup', 'persinc', 'totalwfh'] #NEED TO DO TOTAL WFH
ONE_HOT_ENCODING = ['relationship', 'carlicence', 'studying', 'emptype', 'homesubregion_ASGS', 'homeregion_ASGS']
NUMERIC = []
BINARY = ['sex', 'mbikelicence', 'otherlicence', 'nolicence', 'fulltimework', 'parttimework', 'casualwork', 'anywork']
from sklearn.preprocessing import OrdinalEncoder


# NEED TO DO, THIS IS THE PREPROCESSING
def preprocess_persons(df):
    '''
    Preprocess the persons dataframe
    '''
    df = df.copy()
    
    # Ordinal Encoding
    ordinal_mappings = {
        'agegroup': {'0->4': 0, '5->9': 1, '10->14': 2, '15->19': 3, '20->24': 4, '25->29': 5, '30->34': 6, '35->39': 7, '40->44': 8, '45->49': 9, '50->54': 10, '55->59': 11, '60->64': 12, '65->69': 13, '70->74': 14, '75->79': 15, '80->84': 16, '85->89': 17, '90->94': 18, '95->99': 19, '100+': 20},
        'persinc': {'Negative income': 0,
                    'Nil income': 0,                       # treat Nil the same as Negative
                    '$1-$149 ($1-$7,799)': 1,
                    '$150-$299 ($7,800-$15,599)': 2,
                    '$300-$399 ($15,600-$20,799)': 3,
                    '$400-$499 ($20,800-$25,999)': 4,
                    '$500-$649 ($26,000-$33,799)': 5,
                    '$650-$799 ($33,800-$41,599)': 6,
                    '$800-$999 ($41,600-$51,999)': 7,
                    '$1,000-$1,249 ($52,000-$64,999)': 8,
                    '$1,250-$1,499 ($65,000-$77,999)': 9,
                    '$1,500-$1,749 ($78,000-$90,999)': 10,
                    '$1,750-$1,999 ($91,000-$103,999)': 11,
                    '$2,000-$2,999 ($104,000-$155,999)': 12,
                    '$3,000-$3,499 ($156,000-$181,999)': 13
                    },
        'totalwfh': {'Never': 0, 'Occasional': 1, 'Frequent': 2, 'Always': 3},
    }
    
    for col in ordinal_mappings:
        enc = OrdinalEncoder(categories=[list(ordinal_mappings[col].keys())])
        if col in df.columns:
            df[col] = enc.fit_transform(df[[col]])
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=ONE_HOT_ENCODING, drop_first=True)
    
    # Binary Encoding
    for col in BINARY:
        if col in df.columns:
            if col == 'sex':
                df[col] = df[col].map({'Male': 1, 'Female': 0})
            else:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
                df[col] = df[col].fillna(0)  # Treat NaN as 'No' (0)
    
    # Numeric columns - ensure they are numeric
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def quick_data():
    df = fetch_data()
    result = preprocess_persons(df)
    result = result.merge(compute_travel_efficiency(), on='persid', how='left')
    result = result.merge(fetch_transport_mode(), on='persid', how='left')
    result = result.dropna(subset=['most_used_mode', 'overall_trip_efficiency'])  # drop rows where target is NaN
    return result


if __name__ == "__main__":
    result = quick_data()
    print(result.head())
    print("Data shape:", result.shape)
    for col in result.columns:
        print("Nan in", col, ":", result[col].isna().sum())