from datetime import datetime, timedelta
from typing import List, Tuple, Optional

def load_metar_file(filepath: str) -> List[Tuple[datetime, str]]:
    """
    Load METAR data from a file.
    Each line in the file is expected to start with a timestamp in the format YYYYMMDDHHMM followed by the METAR report text.
    Parameters:
    - filepath (str): Path to the METAR data file.
    Returns:
    - List[Tuple[datetime, str]]: A list of tuples containing the report time and the corresponding METAR report text, sorted by report time.
    """

    with open(filepath, 'r') as f:
        lines = f.readlines()
    metars = [(datetime.strptime(line[:12], "%Y%m%d%H%M"), line.strip()) for line in lines if line.strip()]
    return sorted(metars)

def load_taf_file(filepath: str) -> List[Tuple[datetime, datetime, datetime, str]]:
    """
    Load TAF data from a file.
    Each line in the file is expected to start with a timestamp in the format YYYYMMDD HHMM followed by the TAF report text.
    The TAF report text contains validity periods in the format DDHH/DDHH.
    Parameters:
    - filepath (str): Path to the TAF data file.
    Returns:
    - List[Tuple[datetime, datetime, datetime, str]]: A list of tuples containing valid from, valid to, issue time, and the corresponding TAF report text.
    """
    tafs = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            issue_time = datetime.strptime(line[:12], "%Y%m%d%H%M")
            full_text = line[13:].strip("= \n")
            parts = full_text.split()

            for token in parts:
                if len(token) == 9 and "/" in token and token.count("/") == 1:
                    from_day = int(token[:2])
                    from_hour = int(token[2:4])
                    to_day = int(token[5:7])
                    to_hour = int(token[7:9])

                    # Construct base date for FROM
                    base_date = datetime(issue_time.year, issue_time.month, 1)

                    valid_from = base_date.replace(day=1) + timedelta(days=from_day - 1, hours=from_hour)

                    # Handle hour = 24 rollover for TO
                    if to_hour == 24:
                        to_hour = 0
                        to_day += 1

                    valid_to = base_date.replace(day=1) + timedelta(days=to_day - 1, hours=to_hour)

                    # Handle potential month rollover
                    if valid_to < valid_from:
                        valid_to += timedelta(days=31)  # just enough to push into next month
                        valid_to = valid_to.replace(day=to_day, hour=to_hour)

                    tafs.append((valid_from, valid_to, issue_time, full_text))
                    break
        except Exception as e:
            print(f"Error parsing line: {line[:30]}... -> {e}")
    return tafs

def get_active_metar_taf(current_time, metars, tafs):
    """
    Get the current active METAR and TAF reports based on the provided current time.
    Parameters:
    - current_time (datetime): The current time to check against.
    - metars (List[Tuple[datetime, str]]): A list of tuples containing report time and METAR text.
    - tafs (List[Tuple[datetime, datetime, datetime, str]]): A list of tuples containing valid from, valid to, issue time, and TAF text.
    Returns:
    - active_metar (Optional[str]): The active METAR report text or None if not found.
    - active_taf (Optional[str]): The active TAF report text prefixed with issue
    """

    # Find Current Active METAR
    active_metar = None
    for report_time, text in reversed(metars):
        if report_time <= current_time:
            active_metar = text
            break

    # Find most recently issued TAF that is valid and issued before current_time
    active_taf = None
    latest_issue_time = None

    for valid_from, valid_to, issue_time, text in tafs:
        if valid_from <= current_time <= valid_to and issue_time <= current_time:
            if latest_issue_time is None or issue_time > latest_issue_time:
                latest_issue_time = issue_time
                active_taf = text
                issue_time_str = issue_time.strftime("%Y%m%d%H%M")
                active_taf = issue_time_str + " " + active_taf

    return active_metar, active_taf