import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import connectorx as cx
from preprocess.atfm.preprocess import preprocess

def get_data(db_url, id_tab, ADSB_tab, arrival_flag, alt_col, max_date, min_date):

    # Get Flight IDs
    query_ids = f"SELECT DISTINCT id FROM {id_tab} WHERE arrival={arrival_flag} AND date BETWEEN {min_date} AND {max_date} AND dep_airport_id != dest_airport_id;"
    ids = cx.read_sql(db_url, query_ids, return_type="arrow")
    ids = ids.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates().values.T.tolist()[0]

    # Get ADSB data of the selected flight IDs
    query_ADSB = f"SELECT time, flight_id, lat, lon, {alt_col} FROM {ADSB_tab} WHERE flight_id IN ({', '.join(map(str, ids))}) AND lat BETWEEN 36.6 AND 37.9 AND lon BETWEEN 125.1 AND 127.5;"
    ADSB = cx.read_sql(db_url, query_ADSB, return_type="arrow")
    ADSB = ADSB.to_pandas(split_blocks=False, date_as_object=False).dropna()

    # Check max min time of the data
    max_time = ADSB['time'].max()
    min_time = ADSB['time'].min()

    # Convert time to datetime
    max_time = pd.to_datetime(max_time, unit='s')
    min_time = pd.to_datetime(min_time, unit='s')

    print(f"Data from {min_time} to {max_time}")

    ADSB['time'] = pd.to_datetime(ADSB['time'], unit='s')
    ADSB = ADSB.groupby('flight_id')

    return ADSB, ids

# Query parameters
db_url = 'YOUR DATABASE URL'
id_tab, ADSB_tab = 'flight', 'trajectory'
max_date, min_date = '20221231', '20220101'

# Filtering parameters
icn_lat, icn_lon, icn_alt = 37.463333, 126.440002, 8.0 # degrees, degrees, meters
min_alt_change, FAF, app_sec_rad = 610, 480, 5 # meters
max_cutoff_range = 120 # kilometers
t = 5 # downsample rate
alt_col = 'baroaltitude' # column name for altitude option: 'baroaltitude' or 'geoaltitude'

arr_dep = []
true_ids_arrival = []
for arrival in [True, False]:
    ADSB, _ = get_data(db_url, id_tab, ADSB_tab, int(arrival), alt_col, max_date, min_date)

    # Get only 10 flights for testing
    ADSB = list(ADSB)
    flt_df_lst = []

    for i, flt_df in tqdm(ADSB, total=len(ADSB), desc="Processing ADSB", ncols=100):
        # Dataframe Adjustment
        flt_df = flt_df.set_index('time')
        flt_df = flt_df.sort_index()
        flt_id = flt_df['flight_id'].iloc[0]

        # Preprocess the trajectory data
        flt_df = preprocess(flt_df, icn_lat, icn_lon, icn_alt, max_cutoff_range, alt_col)
        if flt_df is None:
            continue

        # Change  index to number and time to column
        flt_df['time'] = flt_df.index
        flt_df['time'] = flt_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        flt_df = flt_df.reset_index(drop=True)

        # Add id column
        flt_df['id'] = flt_id

        # first time of apprearance
        first_time = flt_df['time'].iloc[0]

        # Add data to list
        flt_df_lst.append(flt_df)
        if arrival:
            true_ids_arrival.append([flt_id, first_time])


    # Plot the trajectory data
    plt.figure(figsize=(10, 10))
    for flt_df in flt_df_lst:
        plt.plot(flt_df['x'], flt_df['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory Data')
    plt.show()
    plt.close()

    all_flt_df = pd.concat(flt_df_lst)
    arr_dep.append(all_flt_df)

# Save the trajectory data
all_adsb = pd.concat(arr_dep)
all_adsb.to_csv('data/airspace_traj_2022.csv', index=False)

# Save the arrival IDs
true_ids_arrival = pd.DataFrame(true_ids_arrival, columns=['id', 'actual_entry_time'])

true_ids_arrival.to_csv('data/arrival_ids_2022.csv', index=False)
