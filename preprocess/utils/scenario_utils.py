import math
from model.atscc import atscc_encoder
from datetime import datetime, timedelta

def describe_aircraft(flight):
    """
    The synthesized aircraft type in our implementation ends with the letter "E" when the type,
    registration number, and wake turbulence category cannot be clearly identified.
    This function checks for that condition and returns an appropriate description.
    Parameters:
        flight: An object containing flight details, including aircraft_type, aircraft_registration, and wake_turbulence_cat.
    Returns:
        A string describing the aircraft details.
    """
    if str(flight.aircraft_type).lower().endswith("e"):
        return (
            "The aircraft type, registration number, and wake turbulence category could not be clearly identified."
        )
    else:
        return (
            f"The aircraft was a {flight.aircraft_type} with registration {flight.aircraft_registration}, "
            f"and it belonged to the {flight.wake_turbulence_cat} wake turbulence category."
        )

def prompt_generator(flight, prompt_format_number):
    """
    Generate a shuffled natural language description of a flight using all 24 input features.
    Parameters:
        flight: An object containing flight details with various attributes.
        prompt_format_number: An integer (1-10) indicating which prompt format to use.
    Returns:
        A string containing the generated flight description.
    """
    if prompt_format_number == 1:
        return (
            f"Scheduled for arrival at {flight.sched_time_utc} UTC on {flight.date} ({flight.day_of_week}), \n"
            f"flight {flight.callsign_code_iata}/{flight.callsign_code_icao} by {flight.airline_name_english} \n"
            f"was set to land at {flight.dest_name_english} ({flight.dest_code_iata}/{flight.dest_code_icao}, \n"
            f"lat: {flight.dest_lat}, lon: {flight.dest_lon}, alt: {flight.dest_altitude} ft). \n"
            f"The aircraft originated from {flight.dep_name_english} ({flight.dep_code_iata}/{flight.dep_code_icao}, \n"
            f"lat: {flight.dep_lat}, lon: {flight.dep_lon}, alt: {flight.dep_altitude} ft), and the total route spanned {flight.distance:.2f} km. \n"
            f"{describe_aircraft(flight)} It was expected to enter airspace or appear via ADS-B at {flight.actual_entry_time}. \n"
            f"This was a {flight.haul}-haul flight."
        )

    elif prompt_format_number == 2:
        return (
            f"On {flight.date}, a {flight.haul}-haul flight by {flight.airline_name_english} was scheduled to arrive at \n"
            f"{flight.dest_name_english} ({flight.dest_code_icao}/{flight.dest_code_iata}) at {flight.sched_time_utc} UTC. \n"
            f"Flight ID: {flight.callsign_code_icao}/{flight.callsign_code_iata}. The aircraft departed from \n"
            f"{flight.dep_name_english} ({flight.dep_code_icao}, lat: {flight.dep_lat}, lon: {flight.dep_lon}, alt: {flight.dep_altitude} ft) and was destined for \n"
            f"{flight.dest_name_english}, located at lat: {flight.dest_lat}, lon: {flight.dest_lon}, alt: {flight.dest_altitude} ft. \n"
            f"{describe_aircraft(flight)} The aircraft was expected to enter monitored airspace around {flight.actual_entry_time}. \n"
            f"Day of week: {flight.day_of_week}, distance: {flight.distance:.2f} km."
        )

    elif prompt_format_number == 3:
        return (
            f"Flight {flight.callsign_code_iata}/{flight.callsign_code_icao} by {flight.airline_name_english} was scheduled to land at \n"
            f"{flight.dest_name_english} ({flight.dest_code_iata}) on {flight.date} ({flight.day_of_week}). \n"
            f"It originated from {flight.dep_name_english} ({flight.dep_code_iata}), located at lat: {flight.dep_lat}, lon: {flight.dep_lon}, alt: {flight.dep_altitude} ft. \n"
            f"{describe_aircraft(flight)} The plane would fly {flight.distance:.2f} km to its destination \n"
            f"at lat: {flight.dest_lat}, lon: {flight.dest_lon}, alt: {flight.dest_altitude} ft. \n"
            f"ADS-B contact or airspace entry was expected at {flight.actual_entry_time}. Haul type: {flight.haul}."
        )

    elif prompt_format_number == 4:
        return (
            f"Expected airspace entry time for flight {flight.callsign_code_icao} was {flight.actual_entry_time}. \n"
            f"This {flight.haul}-haul flight operated by {flight.airline_name_english} was scheduled to arrive at {flight.sched_time_utc} UTC \n"
            f"on {flight.date} ({flight.day_of_week}). It originated from {flight.dep_name_english} ({flight.dep_code_icao}/{flight.dep_code_iata}, \n"
            f"lat: {flight.dep_lat}, lon: {flight.dep_lon}, alt: {flight.dep_altitude} ft) and was headed for \n"
            f"{flight.dest_name_english} ({flight.dest_code_icao}/{flight.dest_code_iata}, lat: {flight.dest_lat}, lon: {flight.dest_lon}, alt: {flight.dest_altitude} ft). \n"
            f"{describe_aircraft(flight)} Total route distance: {flight.distance:.2f} km."
        )

    elif prompt_format_number == 5:
        return (
            f"A {flight.haul}-haul route spanning {flight.distance:.2f} km was scheduled for flight {flight.callsign_code_icao} \n"
            f"({flight.callsign_code_iata}) by {flight.airline_name_english} on {flight.date} ({flight.day_of_week}). \n"
            f"The flight departed from {flight.dep_name_english} ({flight.dep_code_iata}), lat/lon: ({flight.dep_lat}, {flight.dep_lon}), alt: {flight.dep_altitude} ft, \n"
            f"and was expected to land at {flight.dest_name_english} ({flight.dest_code_icao}), lat/lon: ({flight.dest_lat}, {flight.dest_lon}), alt: {flight.dest_altitude} ft. \n"
            f"Arrival time was set for {flight.sched_time_utc} UTC. {describe_aircraft(flight)} Entry into monitored airspace was expected at {flight.actual_entry_time}."
        )

    elif prompt_format_number == 6:
        return (
            f"Flight {flight.callsign_code_icao} operated by {flight.airline_name_english}, with IATA code {flight.callsign_code_iata}, \n"
            f"was en route from {flight.dep_name_english} ({flight.dep_code_icao}) to {flight.dest_name_english} ({flight.dest_code_icao}). \n"
            f"Departure location: lat {flight.dep_lat}, lon {flight.dep_lon}, alt {flight.dep_altitude} ft. \n"
            f"Arrival location: lat {flight.dest_lat}, lon {flight.dest_lon}, alt {flight.dest_altitude} ft. \n"
            f"Scheduled arrival on {flight.date} ({flight.day_of_week}) at {flight.sched_time_utc} UTC. \n"
            f"{describe_aircraft(flight)} Airspace entry estimated at {flight.actual_entry_time}, haul: {flight.haul}, distance: {flight.distance:.2f} km."
        )

    elif prompt_format_number == 7:
        return (
            f"On {flight.date} ({flight.day_of_week}), {flight.airline_name_english} operated flight {flight.callsign_code_icao} \n"
            f"({flight.callsign_code_iata}), scheduled to land at {flight.dest_name_english} ({flight.dest_code_icao}) \n"
            f"at {flight.sched_time_utc} UTC. The origin was {flight.dep_name_english} ({flight.dep_code_icao}), \n"
            f"positioned at lat: {flight.dep_lat}, lon: {flight.dep_lon}, alt: {flight.dep_altitude} ft. \n"
            f"{describe_aircraft(flight)} Destination coordinates: lat {flight.dest_lat}, lon {flight.dest_lon}, alt {flight.dest_altitude} ft. \n"
            f"The plane was expected to enter monitored airspace around {flight.actual_entry_time} and cover {flight.distance:.2f} km. Type: {flight.haul}."
        )

    elif prompt_format_number == 8:
        return (
            f"{flight.airline_name_english} scheduled flight {flight.callsign_code_icao}/{flight.callsign_code_iata} \n"
            f"to arrive in Korea on {flight.date} ({flight.day_of_week}). Expected arrival time: {flight.sched_time_utc} UTC. \n"
            f"The aircraft would depart from {flight.dep_name_english} ({flight.dep_code_iata}, lat: {flight.dep_lat}, lon: {flight.dep_lon}, alt: {flight.dep_altitude} ft) \n"
            f"and land at {flight.dest_name_english} ({flight.dest_code_iata}, lat: {flight.dest_lat}, lon: {flight.dest_lon}, alt: {flight.dest_altitude} ft). \n"
            f"{describe_aircraft(flight)} The distance flown was {flight.distance:.2f} km, and entry into airspace was expected at {flight.actual_entry_time}. Flight type: {flight.haul}."
        )

    elif prompt_format_number == 9:
        return (
            f"Arriving at {flight.dest_name_english} on {flight.date} ({flight.day_of_week}), flight {flight.callsign_code_icao} \n"
            f"by {flight.airline_name_english} was scheduled for {flight.sched_time_utc} UTC. It flew from {flight.dep_name_english} \n"
            f"({flight.dep_code_icao}/{flight.dep_code_iata}), lat: {flight.dep_lat}, lon: {flight.dep_lon}, alt: {flight.dep_altitude} ft, \n"
            f"to destination lat: {flight.dest_lat}, lon: {flight.dest_lon}, alt: {flight.dest_altitude} ft. \n"
            f"{describe_aircraft(flight)} The aircraft was predicted to appear on radar at {flight.actual_entry_time}. Distance: {flight.distance:.2f} km. Category: {flight.haul}-haul."
        )

    elif prompt_format_number == 10:
        return (
            f"On {flight.date}, {flight.airline_name_english} planned flight {flight.callsign_code_icao} ({flight.callsign_code_iata}) \n"
            f"to land at {flight.dest_name_english} ({flight.dest_code_iata}) at {flight.sched_time_utc} UTC. \n"
            f"The aircraft would depart from {flight.dep_name_english} ({flight.dep_code_iata}), coordinates: ({flight.dep_lat}, {flight.dep_lon}), alt: {flight.dep_altitude} ft, \n"
            f"and arrive at destination ({flight.dest_lat}, {flight.dest_lon}, alt: {flight.dest_altitude} ft). \n"
            f"{describe_aircraft(flight)} Estimated airspace entry time: {flight.actual_entry_time}. Flight distance: {flight.distance:.2f} km. Haul class: {flight.haul}."
        )

    else:
        return "Invalid prompt format number."

def load_atscc_encoder(hf_token, device):
    """
    Use this function to load the pre-trained ATSCC encoder model.
    Parameters:
        hf_token: Hugging Face authentication token for accessing the model.
        device: The device (CPU or GPU) to load the model onto.
    Returns:
        The loaded ATSCC encoder model.
    """
    model_args = {
        'input_dims': 9,
        'output_dims': 320,
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'hidden_dim': 3072,
        'dropout': 0.35,
        'random_mask_prob': 0.2
    }
    Encoder = atscc_encoder.from_pretrained("petchthwr/atscc_rksi_85M", **model_args, use_auth_token=hf_token)
    Encoder.to(device)
    Encoder.eval()
    return Encoder


def check_missing_data(df):
    """
    Check for missing data in each column of the DataFrame and print the count of missing values.
    Parameters:
        df : pandas DataFrame to check for missing data
    Returns:
        None
    """
    missing_data = df.isnull().sum()
    for col, missing_count in missing_data.items():
        if missing_count > 0:
            print(f"Column '{col}' has {missing_count} missing values.")


def parse_kst_datetime(row, time_col):
    """
    Convert date and time in KST to UTC datetime.
    Parameters:
        row : DataFrame row containing 'date' and time column
        time_col : Name of the time column ('sched_time' or 'act_time')
    Returns:
        UTC datetime object
    """
    time_str = f"{row['date']}{row[time_col]:0>4}"  # zero-pad to HHMM
    dt_kst = datetime.strptime(time_str, "%Y%m%d%H%M")  # naive
    dt_utc = dt_kst - timedelta(hours=9)  # convert to UTC
    return dt_utc


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two point on the Earth specified in decimal degrees using the Haversine formula.
    Parameters:
        lat1, lon1 : Latitude and Longitude of point 1 (in decimal degrees)
        lat2, lon2 : Latitude and Longitude of point 2 (in decimal degrees)
    Returns:
        Distance between the two points in kilometers.
    """
    R = 6371  # Earth radius in kilometers
    # Convert degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    # Haversine formula
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
