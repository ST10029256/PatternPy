import pandas as pd

def generate_sample_df_with_pattern(pattern: str) -> pd.DataFrame:
    """
    Returns a small DataFrame with 'date', 'Open', 'High', 'Low', 'Close', 'Volume'
    that attempts to showcase the requested pattern (e.g., 'Head and Shoulder').

    Parameters:
        pattern (str): Name of the pattern to mock. 
                       Example: 'Head and Shoulder', 'Inverse Head and Shoulder', 
                                'Double Top', 'Double Bottom', etc.
    """
    date_rng = pd.date_range(start='1/1/2020', end='1/10/2020', freq='D')
    data = {'date': date_rng}

    # Simple “template” data. 
    # Adjust or expand for more patterns as needed.
    if pattern == 'Head and Shoulder':
        data['Open'] =  [90, 85, 80, 90, 85, 80, 75, 80, 85, 90]
        data['High'] =  [95, 90, 85, 95, 90, 85, 80, 85, 90, 95]
        data['Low'] =   [80, 75, 70, 80, 75, 70, 65, 70, 75, 80]
        data['Close'] = [90, 85, 80, 90, 85, 80, 75, 80, 85, 90]
        data['Volume'] = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]

    elif pattern == 'Inverse Head and Shoulder':
        data['Open'] =  [20, 25, 30, 20, 25, 30, 35, 30, 25, 20]
        data['High'] =  [25, 30, 35, 25, 30, 35, 40, 35, 30, 25]
        data['Low'] =   [15, 20, 25, 15, 20, 25, 30, 25, 20, 15]
        data['Close'] = [20, 25, 30, 20, 25, 30, 35, 30, 25, 20]
        data['Volume'] = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]

    elif pattern in ("Double Top", "Double Bottom", "Ascending Triangle", "Descending Triangle"):
        # Minimal example data. You can expand to emulate the pattern more precisely.
        # For demonstration, we just manipulate some rows to create visible lumps/dips.
        data['Open'] =  [80, 75, 70, 80, 75, 70, 65, 70, 75, 80]
        data['High'] =  [95, 90, 85, 95, 90, 85, 80, 85, 90, 95]
        data['Low'] =   [70, 65, 60, 70, 65, 60, 55, 60, 65, 70]
        data['Close'] = [85, 80, 75, 85, 80, 75, 70, 75, 80, 85]
        data['Volume'] = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
        # You could tweak rows 3:5 or 6:8, etc. for bigger pattern spikes if desired.

    else:
        # Default fallback if pattern not recognized
        data['Open'] =  [10, 11, 12, 11, 10, 11, 12, 11, 10, 11]
        data['High'] =  [12, 12, 13, 12, 11, 12, 13, 12, 11, 12]
        data['Low'] =   [9, 10, 11, 10, 9, 10, 11, 10, 9, 10]
        data['Close'] = [11, 11, 12, 11, 10, 11, 12, 11, 10, 11]
        data['Volume'] = [1000]*10

    df = pd.DataFrame(data)
    df.rename(columns={'date': 'Date'}, inplace=True)
    return df
