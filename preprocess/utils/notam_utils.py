import csv
import re
import pickle
import numpy as np
from datetime import datetime

def flatten_notam_csv(input_csv_path, output_txt_path):
    """
    Flatten NOTAM CSV file into a one-line text format.
    Each line in the output text file corresponds to a NOTAM entry with fields:
    issue time, Fir, Location, Qcode, Notam#, Start Date UTC, End
    Date UTC, Full Text - all concatenated into a single line.
    ️ Args:
        input_csv_path (str): Path to the input NOTAM CSV file.
        output_txt_path (str): Path to the output flattened NOTAM text file.
    Returns:
        None
    Example:
        notam_csv_path = 'data/notam.csv'
        notam_txt_path = 'data/notam.txt'
        flatten_notam_csv(notam_csv_path, notam_txt_path)
    """

    with open(input_csv_path, 'r', encoding='utf-8') as infile, \
         open(output_txt_path, 'w', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        for row in reader:
            issue = row['\ufeffissue time']
            fir = row['Fir']
            loc = row['Location']
            qcode = row['Qcode']
            notam_no = row['Notam#']
            start = row['Start Date UTC']
            end = row['End Date UTC']
            text = row['Full Text']

            # Clean newlines and excess whitespace
            flat_text = re.sub(r'\s+', ' ', text).strip()

            # Compose the one-line entry
            line = f"{issue} {fir} {loc} {qcode} {notam_no} {start} {end} {flat_text}"
            outfile.write(line + '\n')

def load_flat_notams(notam_txt_path):
    """
    Load Flattened NOTAMs from a text file.
    Each line in the text file corresponds to a NOTAM entry with fields:
    issue time, Fir, Location, Qcode, Notam#, Start Date UTC, End
    Date UTC, Full Text - all concatenated into a single line.
    Args:
        notam_txt_path (str): Path to the flattened NOTAM text file.
    Returns:
        list of dict: List of NOTAM entries with keys:
            'issue_time', 'fir', 'location', 'qcode', 'notam_id',
            'start', 'end', 'text'.
    Example:
        notam_txt_path = 'data/notam.txt'
        notams = load_flat_notams(notam_txt_path)
    """
    notams = []
    with open(notam_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 7)
            if len(parts) < 8:
                continue  # Skip malformed lines

            notams.append({
                'issue_time': parts[0] + ' ' + parts[1],
                'fir': parts[2],
                'location': parts[3],
                'qcode': parts[4],
                'notam_id': parts[5],
                'start': parts[6],
                'end': parts[7].split(' ', 1)[0],
                'text': parts[7].split(' ', 1)[1] if ' ' in parts[7] else '',
            })
    return notams

def query_active_notams(notams, current_time):
    """
    Query active NOTAMs at the given current time.
    Args:
        notams (list of dict): List of NOTAM entries.
        current_time (datetime or str): Current time to check for active NOTAMs.
            If str, should be in format '%y%m%d%H%M' or '%Y-%m-%d %H:%M:%S'.
    Returns:
        list of dict: List of active NOTAM entries at the given time.
    Example:
        notams = load_flat_notams('data/notam.txt')
        current_time = datetime.strptime('2022-01-15 12:00:00', '%Y-%m-%d %H:%M:%S')
        active_notams = query_active_notams(notams, current_time)

    """
    if isinstance(current_time, str):
        for fmt in ("%y%m%d%H%M", "%Y-%m-%d %H:%M:%S"):
            try:
                current_time = datetime.strptime(current_time, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unsupported time format: {current_time}")

    elif not isinstance(current_time, datetime):
        raise TypeError("current_time must be a datetime object or valid string")

    active = []
    for notam in notams:
        try:
            start_dt = datetime.strptime(notam['start'], "%y%m%d%H%M")
            end_dt = datetime.strptime(notam['end'], "%y%m%d%H%M")
            if start_dt <= current_time <= end_dt:
                active.append(notam)
        except Exception as e:
            print(f"⚠️ Skipping NOTAM [{notam.get('notam_id', 'UNKNOWN')}]: {e}")
    return active

def integrate_notam(data, notam_path):
    """
    Integrate active NOTAMs into scenario data based on current time.
    Args:
        data (list of dict): List of scenario data entries.
        notam_path (str): Path to the flattened NOTAM text file.
    Returns:
        None. The function modifies the input data in place by adding
        a 'notam_prompt' field to each scenario with active NOTAM descriptions.
    Example:
        data = load_scenarios('data/scenarios.pkl')
        integrate_notam(data, 'data/notam.txt')
    """
    notams = load_flat_notams(notam_path)

    for scenario in data:

        current_time = datetime.strptime(scenario['current_time'], "%Y-%m-%d %H:%M:%S")
        active_notams = query_active_notams(notams, current_time)

        # Extract text after 'E)' for each NOTAM
        notam_text_list = []
        for n in active_notams:
            full_text = n.get('text', '')
            if 'E)' in full_text:
                desc = full_text.split('E)', 1)[1].strip()
                notam_text_list.append(desc)
            else:
                notam_text_list.append(full_text.strip())  # Fallback if 'E)' missing

        if len(notam_text_list) == 0:
            notam_text = "No active NOTAMs at this current time."
        else:
            notam_text = "Active NOTAMs: " + " ".join(notam_text_list)

        scenario['notam_prompt'] = notam_text

def integrate_dt_dummy(data):
    """
    Integrate dt_dummy into scenario data.
    dt_dummy = Actual Entry Time - Scheduled Arrival Time
    This is used to help predict delay more accurately.
    Args:
        data (list of dict): List of scenario data entries.
    Returns:
        None. The function modifies the input data in place by adding
        a 'dt_dummy' field to each scenario's label.
    Example:
        data = load_scenarios('data/scenarios.pkl')
        integrate_dt_dummy(data)
    """
    for scenario in data:
        # Schedule arrival time
        t_arrive_schedule = scenario['flight_schedule']['sched_time_utc']  # Needed for calculating delay

        # Actual entry time and delay for calculating actual arrival time
        t_entry = scenario['flight_schedule']['actual_entry_time']  # Needed for calculating delay

        t_entry_dt = datetime.strptime(t_entry, "%Y-%m-%d %H:%M:%S")
        t_arrive_dt = datetime.strptime(t_arrive_schedule, "%Y-%m-%d %H:%M:%S")

        # Convert dt to timedelta
        dt_dummy = (t_entry_dt - t_arrive_dt).total_seconds() / 60  # A dummy variable to calculate delay in minutes
        scenario['label']['dt_dummy'] = dt_dummy # For predicted delay = dt_dummy + dt_in_airspace

        delay = scenario['label']['time_spend_in_airspace'] + dt_dummy
        delay_real = scenario['label']['delay_in_mins']

        # Check if the delay is consistent
        if not np.isclose(delay, delay_real, atol=1e-2):
            print(f"Inconsistent delay for flight {scenario['flight_id']}: "
                  f"Predicted delay = {delay}, Actual delay = {delay_real}")

"""
Example usage: Integrate NOTAMs into scenario data files for each month.
num_data_list = []
months = ['01']
for month in months:
    print(f"Postprocessing month {month}")
    data = load_scenarios(f'../data/scenario_generation_2022-{month}.pkl')
    num_data = len(data)
    print(f"Loaded {num_data} scenarios for month {month}")
    num_data_list.append(num_data)

    integrate_dt_dummy(data)
    integrate_notam(data, "../../data/notam.txt")
    
    save_path = f"../data/scenario_generation_2022-{month}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {save_path}")
"""
