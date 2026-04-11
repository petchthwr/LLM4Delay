from preprocess.utils.airspace_utils import load_traj_file, find_active_traj, pad_stack_traj
from preprocess.utils.notam_utils import integrate_notam, integrate_dt_dummy
from preprocess.utils.weather_utils import get_active_metar_taf, load_metar_file, load_taf_file
from preprocess.utils.scenario_utils import load_atscc_encoder, prompt_generator
import random
from datetime import datetime
import pandas as pd
import torch
from tqdm import tqdm
import pickle

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
random.seed(0)

# ENTER YOUR HUGGINGFACE API KEY HERE (for loading ATSCC Encoder)
hf_key = "YOUR HUGGINGFACE TOKEN"

# Raw data loading
metars = load_metar_file("../data/metar_2022.txt")
tafs = load_taf_file("../data/taf_2022.txt")
notams = "../data/notam.txt"
flight_schedule = pd.read_csv('../data/arrival_schedule_2022.csv').dropna(how='any')
traj_data = load_traj_file("../data/airspace_traj_2022.csv")

# Load ATSCC Encoder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Encoder = load_atscc_encoder(hf_key, device)


traj_data_backup = traj_data.copy()
flight_schedule_backup = flight_schedule.copy()
months = ["2022-01"]
for month in months:
    print(f"Processing month {month}...")

    # Reset data to backup
    traj_data = traj_data_backup.copy()
    flight_schedule = flight_schedule_backup.copy()

    data = []
    traj_data['time'] = pd.to_datetime(traj_data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    month_start = pd.Timestamp(f"{month}-01")
    prev_month_start = month_start - pd.DateOffset(days=3)
    next_month_start = month_start + pd.DateOffset(months=1)

    traj_data = traj_data[
        (traj_data['time'] >= prev_month_start) &
        (traj_data['time'] < next_month_start)
    ]
    traj_data['time'] = traj_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    flight_schedule = flight_schedule[pd.to_datetime(flight_schedule['date']).dt.strftime('%Y-%m') == month]
    total_flights = len(flight_schedule)

    for flight in tqdm(flight_schedule.itertuples(index=False), total=total_flights, desc="Processing flights", ncols=100):

        # Prediction Labels
        current_id = flight.id

        actual_time_arrival = flight.act_time_utc
        time_spend_in_airspace = round(flight.airspace_ate_min, 2)
        delay_in_mins = flight.delay_mins
        delay_bool = flight.delay_gt_15min
        label = [actual_time_arrival, time_spend_in_airspace, delay_in_mins, delay_bool]
        label_dict = {'actual_time_arrival': actual_time_arrival, 'time_spend_in_airspace': time_spend_in_airspace, 'delay_in_mins': delay_in_mins, 'delay_bool': delay_bool}

        num_timestamp = len(traj_data[traj_data['id'] == current_id]['time'].unique().tolist())
        current_times = random.sample(traj_data[traj_data['id'] == current_id]['time'].unique().tolist(), min(1, num_timestamp))

        for current_time in current_times:
            try:
                # Create Natural Language prompt
                ## Convert current_time to the format 'YYYY-MM-DD HH:MM:SS'
                current_time_prompt = 'Current time: ' + datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')

                ## Natural language description of the weather
                active_metar, active_taf = get_active_metar_taf(datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S'), metars, tafs)
                weather_prompt = ( f"METAR in effect: {active_metar}\n"
                                   f"TAF in effect: {active_taf}")

                ## Natural language description of the flight
                prompt_format_number = random.randint(1, 10)
                flight_prompt = prompt_generator(flight, prompt_format_number)

                ## Total NLP prompt
                flight_prompt = f"{current_time_prompt}\n{flight_prompt}\n{weather_prompt}"
                # strip \n out of the prompt
                flight_prompt = flight_prompt.replace('\n', ' ')

                focusing_traj, active_traj, affecting_traj = find_active_traj(traj_data, current_time, current_id)
                focusing_traj, active_traj, affecting_traj = pad_stack_traj(focusing_traj, active_traj, affecting_traj) # Numpy array Shape (T_foc, 9), (N_act, T_act, 9), (N_aff, T_aff, 9)

                # Encoding the trajectory data using ATSCC
                torch.cuda.empty_cache()
                with torch.no_grad():
                    z_focusing = Encoder.instance_level_encode(torch.tensor(focusing_traj, dtype=torch.float32).unsqueeze(0).to(device))  # Shape (1, 320)
                    z_active = Encoder.instance_level_encode(torch.tensor(active_traj, dtype=torch.float32).to(device)) if active_traj is not None else None  # Shape (N_act, 320)
                    z_affecting = Encoder.instance_level_encode(torch.tensor(affecting_traj, dtype=torch.float32).to(device)) if affecting_traj is not None else None  # Shape (N_aff, 320)

                # Append the data to the list
                data.append({
                    'current_time': current_time,
                    'flight_id': current_id,
                    'flight_prompt': flight_prompt,
                    'z_focusing': z_focusing.cpu().numpy(),
                    'z_active': z_active.cpu().numpy() if z_active is not None else None,
                    'z_affecting': z_affecting.cpu().numpy() if z_affecting is not None else None,
                    'traj_focusing': focusing_traj, # For older baselines method in ATM
                    'traj_active': active_traj,
                    'traj_affecting': affecting_traj,
                    'flight_schedule': flight._asdict(), # For older baselines method in ATM
                    'label': label_dict
                })
            except:
                continue


    print("\n-------------- Scenario Generation Completed --------------")
    print(f"Total scenarios generated: {len(data)}")

    # Should is save data in pickle format
    data.sort(key=lambda x: x['current_time'])

    print("\n-------------- Integrating additional information --------------")
    integrate_dt_dummy(data) # Integrate the duration of (Airspace Entry Time - Scheduled Arrival Time)
    integrate_notam(data, notams) # Integrate NOTAM prompt. Recently Added!

    # Print first data current time and last data current time
    print(f"First data current time: {data[0]['current_time']}")
    print(f"Last data current time: {data[-1]['current_time']}")

    with open(f'../data/scenario_generation_{month}_deli.pkl', 'wb') as f:

        pickle.dump(data, f)
