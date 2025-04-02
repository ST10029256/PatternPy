import pandas as pd
import numpy as np

def detect_head_shoulder(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Detects standard and inverse head-and-shoulders patterns using a rolling window.

    Parameters:
        df (pd.DataFrame): Must have columns 'High' and 'Low' at minimum.
        window (int): Rolling window size for local max/min detection.
    """
    roll_window = window

    # Rolling max/min of High/Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()

    # Boolean masks for Head & Shoulder vs. Inverse Head & Shoulder
    mask_head_shoulder = (
        (df['high_roll_max'] > df['High'].shift(1)) &
        (df['high_roll_max'] > df['High'].shift(-1)) &
        (df['High'] < df['High'].shift(1)) &
        (df['High'] < df['High'].shift(-1))
    )
    mask_inv_head_shoulder = (
        (df['low_roll_min'] < df['Low'].shift(1)) &
        (df['low_roll_min'] < df['Low'].shift(-1)) &
        (df['Low'] > df['Low'].shift(1)) &
        (df['Low'] > df['Low'].shift(-1))
    )

    # Mark pattern
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'

    return df


def detect_multiple_tops_bottoms(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Detects multiple tops/bottoms within a given rolling window.

    Parameters:
        df (pd.DataFrame): Must have columns 'High', 'Low', 'Close'.
        window (int): Rolling window size.
    """
    roll_window = window

    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['close_roll_max'] = df['Close'].rolling(window=roll_window).max()
    df['close_roll_min'] = df['Close'].rolling(window=roll_window).min()

    # Simple boolean masks for "multiple top" vs. "multiple bottom"
    mask_top = (
        (df['high_roll_max'] >= df['High'].shift(1)) &
        (df['close_roll_max'] < df['Close'].shift(1))
    )
    mask_bottom = (
        (df['low_roll_min'] <= df['Low'].shift(1)) &
        (df['close_roll_min'] > df['Close'].shift(1))
    )

    df['multiple_top_bottom_pattern'] = np.nan
    df.loc[mask_top, 'multiple_top_bottom_pattern'] = 'Multiple Top'
    df.loc[mask_bottom, 'multiple_top_bottom_pattern'] = 'Multiple Bottom'

    return df


def calculate_support_resistance(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Computes basic rolling support/resistance using mean ± 2*std of 'High'/'Low'.
    """
    roll_window = window
    std_dev = 2

    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()

    mean_high = df['High'].rolling(window=roll_window).mean()
    std_high = df['High'].rolling(window=roll_window).std()

    mean_low = df['Low'].rolling(window=roll_window).mean()
    std_low = df['Low'].rolling(window=roll_window).std()

    df['support'] = mean_low - std_dev * std_low
    df['resistance'] = mean_high + std_dev * std_high

    return df


def detect_triangle_pattern(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Very simplistic 'triangle' detection based on rolling extremes.
    """
    roll_window = window

    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()

    mask_asc = (
        (df['high_roll_max'] >= df['High'].shift(1)) &
        (df['low_roll_min'] <= df['Low'].shift(1)) &
        (df['Close'] > df['Close'].shift(1))
    )
    mask_desc = (
        (df['high_roll_max'] <= df['High'].shift(1)) &
        (df['low_roll_min'] >= df['Low'].shift(1)) &
        (df['Close'] < df['Close'].shift(1))
    )

    df['triangle_pattern'] = np.nan
    df.loc[mask_asc, 'triangle_pattern'] = 'Ascending Triangle'
    df.loc[mask_desc, 'triangle_pattern'] = 'Descending Triangle'

    return df


def detect_wedge(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Tries to detect wedge patterns by tracking short-term 'trend_high'/'trend_low'.
    """
    roll_window = window

    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()

    # Lambda: +1 if up slope, -1 if down slope, 0 if flat
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(
        lambda x: 1 if (x.iloc[-1] - x.iloc[0]) > 0 else (-1 if (x.iloc[-1] - x.iloc[0]) < 0 else 0),
        raw=False
    )
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(
        lambda x: 1 if (x.iloc[-1] - x.iloc[0]) > 0 else (-1 if (x.iloc[-1] - x.iloc[0]) < 0 else 0),
        raw=False
    )

    mask_wedge_up = (
        (df['high_roll_max'] >= df['High'].shift(1)) &
        (df['low_roll_min'] <= df['Low'].shift(1)) &
        (df['trend_high'] == 1) & (df['trend_low'] == 1)
    )
    mask_wedge_down = (
        (df['high_roll_max'] <= df['High'].shift(1)) &
        (df['low_roll_min'] >= df['Low'].shift(1)) &
        (df['trend_high'] == -1) & (df['trend_low'] == -1)
    )

    df['wedge_pattern'] = np.nan
    df.loc[mask_wedge_up, 'wedge_pattern'] = 'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = 'Wedge Down'

    return df


# Replace the existing detect_channel function with this one:
def detect_channel(df, window=3):
    # Define the rolling window
    roll_window = window
    # Define a factor to check for the range of channel
    channel_range = 0.1
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()

     # Apply lambda with length check to avoid index error
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(
        lambda x: (1 if (x.iloc[-1] - x.iloc[0]) > 0 else -1 if (x.iloc[-1] - x.iloc[0]) < 0 else 0) if len(x) >= 2 else 0,
        raw=False
    )
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(
        lambda x: (1 if (x.iloc[-1] - x.iloc[0]) > 0 else -1 if (x.iloc[-1] - x.iloc[0]) < 0 else 0) if len(x) >= 2 else 0,
        raw=False
    )

    # Create a boolean mask for Channel Up pattern
    mask_channel_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    # Create a boolean mask for Channel Down pattern
    mask_channel_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    # Create a new column for Channel Up and Channel Down pattern and populate it using the boolean masks
    df['channel_pattern'] = np.nan
    df.loc[mask_channel_up, 'channel_pattern'] = 'Channel Up'
    df.loc[mask_channel_down, 'channel_pattern'] = 'Channel Down'
    return df

def detect_double_top_bottom(df: pd.DataFrame, window: int = 3, threshold: float = 0.05) -> pd.DataFrame:
    """
    Detects Double Top / Double Bottom with a simplistic approach, using a threshold
    to check if subsequent peaks/troughs are within a certain percentage range.

    Parameters:
        window (int): Rolling window to find local min/max.
        threshold (float): Relative range threshold for matching top/bottom heights.
    """
    roll_window = window

    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()

    # Double Top
    mask_double_top = (
        (df['high_roll_max'] >= df['High'].shift(1)) &
        (df['high_roll_max'] >= df['High'].shift(-1)) &
        (df['High'] < df['High'].shift(1)) &
        (df['High'] < df['High'].shift(-1)) &
        ((df['High'].shift(1) - df['Low'].shift(1)) <= threshold * ((df['High'].shift(1) + df['Low'].shift(1))/2)) &
        ((df['High'].shift(-1) - df['Low'].shift(-1)) <= threshold * ((df['High'].shift(-1) + df['Low'].shift(-1))/2))
    )
    # Double Bottom
    mask_double_bottom = (
        (df['low_roll_min'] <= df['Low'].shift(1)) &
        (df['low_roll_min'] <= df['Low'].shift(-1)) &
        (df['Low'] > df['Low'].shift(1)) &
        (df['Low'] > df['Low'].shift(-1)) &
        ((df['High'].shift(1) - df['Low'].shift(1)) <= threshold * ((df['High'].shift(1) + df['Low'].shift(1))/2)) &
        ((df['High'].shift(-1) - df['Low'].shift(-1)) <= threshold * ((df['High'].shift(-1) + df['Low'].shift(-1))/2))
    )

    df['double_pattern'] = np.nan
    df.loc[mask_double_top, 'double_pattern'] = 'Double Top'
    df.loc[mask_double_bottom, 'double_pattern'] = 'Double Bottom'

    return df


def detect_trendline(df: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """
    Example slope + intercept calculation on the last N 'Close' points in a rolling manner.
    Creates columns 'slope' and 'intercept', then 'support'/'resistance' as a naive projection.
    """
    roll_window = window

    df['slope'] = np.nan
    df['intercept'] = np.nan

    for i in range(roll_window, len(df)):
        x = np.array(range(i - roll_window, i))
        y = df['Close'][i - roll_window:i]
        A = np.vstack([x, np.ones(len(x))]).T

        # Solve linear regression via least squares
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        df.at[df.index[i], 'slope'] = m
        df.at[df.index[i], 'intercept'] = c

    mask_support = df['slope'] > 0
    mask_resistance = df['slope'] < 0

    df['support'] = np.nan
    df['resistance'] = np.nan

    df.loc[mask_support, 'support'] = df['Close'] * df['slope'] + df['intercept']
    df.loc[mask_resistance, 'resistance'] = df['Close'] * df['slope'] + df['intercept']

    return df


def find_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example pivot finder that flags HH, LL, LH, HL signals.
    NOTE: Adjust columns to match 'High'/'Low' used elsewhere.
    """
    # Differences between consecutive highs/lows
    high_diffs = df['High'].diff()
    low_diffs = df['Low'].diff()

    # Basic masks
    higher_high_mask = (high_diffs > 0) & (high_diffs.shift(-1) < 0)
    lower_low_mask = (low_diffs < 0) & (low_diffs.shift(-1) > 0)
    lower_high_mask = (high_diffs < 0) & (high_diffs.shift(-1) > 0)
    higher_low_mask = (low_diffs > 0) & (low_diffs.shift(-1) < 0)

    df['signal'] = ''
    df.loc[higher_high_mask, 'signal'] = 'HH'
    df.loc[lower_low_mask, 'signal'] = 'LL'
    df.loc[lower_high_mask, 'signal'] = 'LH'
    df.loc[higher_low_mask, 'signal'] = 'HL'

    return df
