import pandas as pd
import connectorx as cx
import os
import numpy as np

from preprocess.utils.scenario_utils import parse_kst_datetime, haversine

db_url = 'YOUR DATABASE URL'
arrival_schedule = pd.read_csv('data/arrival_ids_2022.csv')
arrival_ids = arrival_schedule['id'].tolist()

# Build SQL query from our database to get flight information
query_text = f"SELECT callsign_id, date, sched_time, act_time, dep_airport_id, dest_airport_id  FROM flight WHERE id IN ({', '.join(map(str, arrival_ids))});"
flight_tab_data = cx.read_sql(db_url, query_text, return_type="arrow")
flight_tab_data = flight_tab_data.to_pandas(split_blocks=False, date_as_object=False)

# Airport Information
# Convert dep_airport_id to dep_airport_info
query_dep_airport_info = f"SELECT id, code_iata, code_icao, name_english, lat, lon, altitude  FROM airport WHERE id IN ({', '.join(map(str, flight_tab_data['dep_airport_id'].tolist()))});"
dep_airport_info = cx.read_sql(db_url, query_dep_airport_info, return_type="arrow").to_pandas(split_blocks=False, date_as_object=False)
dep_airport_info.columns = ['dep_' + col for col in dep_airport_info.columns]
flight_tab_data = flight_tab_data.merge(dep_airport_info, left_on='dep_airport_id', right_on='dep_id', how='left')

# Convert dest_airport_id to dest_airport_info
query_dest_airport_info = f"SELECT id, code_iata, code_icao, name_english, lat, lon, altitude  FROM airport WHERE id IN ({', '.join(map(str, flight_tab_data['dest_airport_id'].tolist()))});"
dest_airport_info = cx.read_sql(db_url, query_dest_airport_info, return_type="arrow").to_pandas(split_blocks=False, date_as_object=False)
dest_airport_info.columns = ['dest_' + col for col in dest_airport_info.columns]
flight_tab_data = flight_tab_data.merge(dest_airport_info, left_on='dest_airport_id', right_on='dest_id', how='left')

# Time Related Information
# Processes Delay times
flight_tab_data['sched_time_utc'] = flight_tab_data.apply(lambda row: parse_kst_datetime(row, 'sched_time'), axis=1)
flight_tab_data['act_time_utc'] = flight_tab_data.apply(lambda row: parse_kst_datetime(row, 'act_time'), axis=1)

# Compute delay in minutes and True/False for delay > 15 minutes
flight_tab_data['delay_mins'] = (flight_tab_data['act_time_utc'] - flight_tab_data['sched_time_utc']).dt.total_seconds() / 60
flight_tab_data['delay_gt_15min'] = flight_tab_data['delay_mins'] > 15

# Find day of week from date
flight_tab_data['date'] = pd.to_datetime(flight_tab_data['date'], format='%Y%m%d')
flight_tab_data['day_of_week'] = flight_tab_data['date'].dt.dayofweek
flight_tab_data['day_of_week'] = flight_tab_data['day_of_week'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})

# Callsign and Carrier Information
# convert callsign_id to callsign_info
callsign_query_text = f"SELECT id, carrier_id, code_iata, code_icao FROM callsign WHERE id IN ({', '.join(map(str, flight_tab_data['callsign_id'].tolist()))});"
callsign_tab_data = cx.read_sql(db_url, callsign_query_text, return_type="arrow").to_pandas(split_blocks=False, date_as_object=False)
callsign_tab_data.columns = ['callsign_' + col for col in callsign_tab_data.columns]
flight_tab_data = flight_tab_data.merge(callsign_tab_data, left_on='callsign_id', right_on='callsign_id', how='left')

# convert carrier_id to airline_info
airline_query_text = f"SELECT id, name_english FROM carrier WHERE id IN ({', '.join(map(str, callsign_tab_data['callsign_carrier_id'].tolist()))});"
airline_tab_data = cx.read_sql(db_url, airline_query_text, return_type="arrow").to_pandas(split_blocks=False, date_as_object=False)
airline_tab_data.columns = ['airline_' + col for col in airline_tab_data.columns]
flight_tab_data = flight_tab_data.merge(airline_tab_data, left_on='callsign_carrier_id', right_on='airline_id', how='left')

# Aircraft Information: Map aircraft type from data/aircraft_data
# create list of df from pd.read_csv for all files in the folder data/aircraft_data
aircraft_tab_data = [pd.read_csv(f'data/aircraft_data/{file}', low_memory=False) for file in os.listdir(
    'data/aircraft_data')]
aircraft_tab_data = pd.concat(aircraft_tab_data)
aircraft_tab_data = aircraft_tab_data[['flight','t', 'r']].dropna()
aircraft_tab_data['flight'] = aircraft_tab_data['flight'].str.strip()

# Map aircraft type to callsign
aircraft_tab_data = aircraft_tab_data.groupby('flight').agg(lambda x:x.value_counts().index[0])
aircraft_tab_data = aircraft_tab_data.reset_index()
aircraft_tab_data.columns = ['callsign_code_icao', 'aircraft_type', 'aircraft_registration']
flight_tab_data = flight_tab_data.merge(aircraft_tab_data, left_on='callsign_code_icao', right_on='callsign_code_icao', how='left')

# Distance and Haul Classification
flight_tab_data['dep_lat'] = flight_tab_data['dep_lat'].astype(float)
flight_tab_data['dep_lon'] = flight_tab_data['dep_lon'].astype(float)
flight_tab_data['dest_lat'] = flight_tab_data['dest_lat'].astype(float)
flight_tab_data['dest_lon'] = flight_tab_data['dest_lon'].astype(float)

flight_tab_data['distance'] = flight_tab_data.apply(lambda row: haversine(row['dep_lat'], row['dep_lon'], row['dest_lat'], row['dest_lon']), axis=1)
flight_tab_data['haul'] = pd.cut(flight_tab_data['distance'], bins=[0, 1600, 4000, float('inf')], labels=['short', 'medium', 'long'])
# If unknown haul use 'local'
flight_tab_data['haul'] = flight_tab_data['haul'].cat.add_categories('local').fillna('local')
# Short-haul
mask_short = (flight_tab_data['haul'] == 'short') & (flight_tab_data['aircraft_type'].isna())
flight_tab_data.loc[mask_short, 'aircraft_type'] = np.random.choice(
    ['A320e', 'B737e'], size=mask_short.sum())
# Medium-haul
mask_medium = (flight_tab_data['haul'] == 'medium') & (flight_tab_data['aircraft_type'].isna())
flight_tab_data.loc[mask_medium, 'aircraft_type'] = np.random.choice(
    ['A320e', 'A330e', 'A340e', 'A350e', 'B737e', 'B777e', 'B767e', 'B787e'], size=mask_medium.sum())
# Long-haul
mask_long = (flight_tab_data['haul'] == 'long') & (flight_tab_data['aircraft_type'].isna())
flight_tab_data.loc[mask_long, 'aircraft_type'] = np.random.choice(
    ['A330e', 'A340e', 'A350e', 'A380e', 'B777e', 'B767e', 'B787e', 'B747e'], size=mask_long.sum())
# Local haul
mask_local = (flight_tab_data['haul'] == 'local') & (flight_tab_data['aircraft_type'].isna())
flight_tab_data.loc[mask_local, 'aircraft_type'] = 'A380' if flight_tab_data.loc[mask_local, 'callsign_code_icao'].str.contains('AAR').any() else np.random.choice(
    ['A320e', 'B737e'], size=mask_local.sum())

# Fill unknown registration with NAXXXX
flight_tab_data['aircraft_registration'] = flight_tab_data['aircraft_registration'].fillna('NODATA')

# Classify wake turbulence category
def classify_wtc(ac_type):
    if isinstance(ac_type, str):
        if ac_type.startswith(('A31', 'A32', 'B73', 'A21N', 'A20N', 'B38M', 'BCS3')):
            return 'medium'
        elif ac_type.startswith(('A33', 'A34', 'A35', 'B75', 'B76', 'B77', 'B78', 'MD11')):
            return 'heavy'
        elif ac_type.startswith(('A38', 'B74')):
            return 'super'
    return None
flight_tab_data['wake_turbulence_cat'] = flight_tab_data['aircraft_type'].apply(classify_wtc)

# Create ATE (Actual time enroute) in airspace (flight_tab_data.actual_time_utc - arrival_schedule.actual_entry_time)
arrival_schedule['actual_entry_time'] = pd.to_datetime(arrival_schedule['actual_entry_time'], format='%Y-%m-%d %H:%M:%S')
flight_tab_data['sched_time_utc'] = pd.to_datetime(flight_tab_data['sched_time_utc'], format='%Y-%m-%d %H:%M:%S')
flight_tab_data['act_time_utc'] = pd.to_datetime(flight_tab_data['act_time_utc'], format='%Y-%m-%d %H:%M:%S')
flight_tab_data['airspace_ate_min'] = (flight_tab_data['act_time_utc'] - arrival_schedule['actual_entry_time']).dt.total_seconds() / 60


# concat dataframes flt_data and flight_plan
flight_schedule = pd.concat([arrival_schedule, flight_tab_data[['airline_name_english', 'callsign_code_iata', 'callsign_code_icao', # Operator information
                                                                'aircraft_type', 'aircraft_registration', 'wake_turbulence_cat', # Aircraft information
                                                                'dep_code_iata', 'dep_code_icao', 'dep_name_english', 'dep_lat', 'dep_lon', 'dep_altitude', # Departure airport information
                                                                'dest_code_iata', 'dest_code_icao', 'dest_name_english', 'dest_lat', 'dest_lon', 'dest_altitude', # Destination airport information
                                                                'haul', 'distance', # Flight information
                                                                'date','day_of_week','sched_time_utc', # Time-related information
                                                                'act_time_utc', 'airspace_ate_min', 'delay_mins', 'delay_gt_15min']]], axis=1) # Time-related information (Label columns)

# Print data type
# print(flight_schedule.dtypes)

# Save the processed data
# flight_schedule.to_csv('data/arrival_schedule_2022.csv', index=False)

"""
# Data Description

# Feature columns
actual_entry_time : Time stamp that first ADS-B transmission appear in the monitor or time that aircraft first entry to the airspace
airline_name_english : Name of the operating airline in english
callsign_code_iata : Callsign in IATA format
callsign_code_icao : Callsign in ICAO format
aircraft_type : Aircaraft Type
aircraft_registration : Aircraft Registration
wake_turbulence_cat : Wake Turbulence Category
dep_code_iata : Departure airport code in IATA format
dep_code_icao : Departure airport code in ICAO format
dep_name_english : Departure airport name
dep_lat : Departure airport latitude
dep_lon : Departure airport longitude
dep_altitude : Departure airport altitute
dest_code_iata : Destination airport code in IATA format
dest_code_icao : Destination airport code in ICAO format
dest_name_english : Destination airport name
dest_lat : Destination airport latitude
dest_lon : Destination airport longitude
dest_altitude : Destination airport altitute
haul : Flight Haul (short, medium, long, local)
distance : Great circle distance between departure and destination airports
date : Date of arrival in Korea Standard Time (KST)
day_of_week : Day of week of arrival in Korea Standard Time (KST)
sched_time_utc : Scheduled time of arrival in UTC

# Label columns
act_time_utc : Actual time of arrival in UTC
airspace_ate_min : Actual time enroute (taken) in airspace in minutes
delay_mins : Delay in minutes (act_time_utc - sched_time_utc)
delay_gt_15min : Delay greater than 15 minutes (True/False)


"""
