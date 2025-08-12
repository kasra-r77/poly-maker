import json
import time
import pandas as pd
import requests
from data_updater.trading_utils import get_clob_client
from data_updater.google_utils import get_spreadsheet
from data_updater.find_markets import add_volatility_to_df
from gspread_dataframe import set_with_dataframe
import traceback
import random
from time import sleep
import numpy as np

def retry_with_backoff(func, *args, max_retries=5, base_delay=1, **kwargs):
    """
    Retry a function with exponential backoff
    
    Args:
        func: The function to retry
        *args: Arguments to pass to the function
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call
    """
    retries = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and retries < max_retries:
                # Rate limited, apply exponential backoff
                sleep_time = base_delay * (2 ** retries) + random.uniform(0, 1)
                print(f"Rate limited. Retrying in {sleep_time:.2f} seconds...")
                sleep(sleep_time)
                retries += 1
            else:
                # Not a rate limit error or max retries exceeded
                raise

def get_bid_ask_range(ret, TICK_SIZE):
    bid_from = ret['midpoint'] - ret['max_spread'] / 100
    bid_to = ret['best_ask'] #Although bid to this high up will change bid_from because of changing midpoint, take optimistic approach

    if bid_to == 0:
        bid_to = ret['midpoint']

    if bid_to - TICK_SIZE > ret['midpoint']:
        bid_to = ret['best_bid'] + (TICK_SIZE + 0.1 * TICK_SIZE)

    if bid_from > bid_to:
        bid_from = bid_to - (TICK_SIZE + 0.1 * TICK_SIZE)

    ask_to = ret['midpoint'] + ret['max_spread'] / 100
    ask_from = ret['best_bid']

    if ask_from == 0:
        ask_from = ret['midpoint']

    if ask_from + TICK_SIZE < ret['midpoint']:
        ask_from = ret['best_ask'] - (TICK_SIZE + 0.1 * TICK_SIZE)

    if ask_from > ask_to:
        ask_to = ask_from + (TICK_SIZE + 0.1 * TICK_SIZE)

    bid_from = round(bid_from, 3)
    bid_to = round(bid_to, 3)
    ask_from = round(ask_from, 3)
    ask_to = round(ask_to, 3)

    if bid_from < 0:
        bid_from = 0

    if ask_from < 0:
        ask_from = 0
        
    return bid_from, bid_to, ask_from, ask_to

def generate_numbers(start, end, TICK_SIZE):
    # Calculate the starting point, rounding up to the next hundredth if not an exact multiple of TICK_SIZE
    rounded_start = (int(start * 100) + 1) / 100 if start * 100 % 1 != 0 else start + TICK_SIZE
    
    # Calculate the ending point, rounding down to the nearest hundredth
    rounded_end = int(end * 100) / 100
    
    # Generate numbers from rounded_start to rounded_end, ensuring they fall strictly within the original bounds
    numbers = []
    current = rounded_start
    while current < end:
        numbers.append(current)
        current += TICK_SIZE
        current = round(current, len(str(TICK_SIZE).split('.')[1]))  # Rounding to avoid floating point imprecision

    return numbers

def add_formula_params(curr_df, midpoint, v, daily_reward):
    curr_df['s'] = (curr_df['price'] - midpoint).abs()
    curr_df['S'] = ((v - curr_df['s']) / v) ** 2
    curr_df['100'] = 1/curr_df['price'] * 100

    curr_df['size'] = curr_df['size'] + curr_df['100']

    curr_df['Q'] = curr_df['S'] * curr_df['size']
    curr_df['reward_per_100'] = (curr_df['Q'] / curr_df['Q'].sum()) * daily_reward / 2 / curr_df['size'] * curr_df['100']
    return curr_df

def process_single_row(row, client):
    """
    Process a single market row with detailed error handling
    """
    # Add a small delay to avoid rate limiting
    sleep(random.uniform(0.2, 0.5))
    
    # Debug: Print the keys in the row to see what we're working with
    print(f"Processing row with keys: {list(row.keys())}")
    
    ret = {}
    
    # Check if required fields exist
    if 'question' not in row:
        print("Error: 'question' field missing in market data")
        return None
    
    ret['question'] = row['question']
    
    if 'neg_risk' not in row:
        print("Warning: 'neg_risk' field missing in market data, using default value")
        ret['neg_risk'] = 0
    else:
        ret['neg_risk'] = row['neg_risk']

    # Check if tokens field exists and has the expected structure
    if 'tokens' not in row:
        print("Error: 'tokens' field missing in market data")
        return None
    
    tokens = row['tokens']
    if not tokens or not isinstance(tokens, list) or len(tokens) < 2:
        print(f"Error: 'tokens' field has unexpected structure: {tokens}")
        return None
    
    # Debug: Print the tokens structure
    print(f"Tokens structure: {tokens}")
    
    # Check if tokens have the expected fields
    if 'outcome' not in tokens[0] or 'outcome' not in tokens[1]:
        print("Error: 'outcome' field missing in tokens")
        return None
    
    ret['answer1'] = tokens[0]['outcome']
    ret['answer2'] = tokens[1]['outcome']
    
    if 'token_id' not in tokens[0] or 'token_id' not in tokens[1]:
        print("Error: 'token_id' field missing in tokens")
        return None
    
    token1 = tokens[0]['token_id']
    token2 = tokens[1]['token_id']
    
    # Check if rewards field exists and has the expected structure
    if 'rewards' not in row:
        print("Error: 'rewards' field missing in market data")
        return None
    
    rewards = row['rewards']
    if not isinstance(rewards, dict):
        print(f"Error: 'rewards' field has unexpected structure: {rewards}")
        return None
    
    # Debug: Print the rewards structure in more detail
    print(f"Rewards structure: {rewards}")
    if 'rates' in rewards:
        print(f"Rewards rates: {rewards['rates']}")
    
    if 'min_size' not in rewards:
        print("Warning: 'min_size' field missing in rewards, using default value")
        ret['min_size'] = 1
    else:
        ret['min_size'] = rewards['min_size']
    
    if 'max_spread' not in rewards:
        print("Warning: 'max_spread' field missing in rewards, using default value")
        ret['max_spread'] = 5
    else:
        ret['max_spread'] = rewards['max_spread']
    
    rate = 0
    if 'rates' in rewards and isinstance(rewards['rates'], list):
        for rate_info in rewards['rates']:
            if isinstance(rate_info, dict) and 'asset_address' in rate_info and 'rewards_daily_rate' in rate_info:
                if rate_info['asset_address'].lower() == '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174'.lower():
                    rate = rate_info['rewards_daily_rate']
                    break
    
    ret['rewards_daily_rate'] = rate
    
    # Use retry with backoff for API calls
    try:
        print(f"Fetching order book for token: {token1}")
        book = retry_with_backoff(client.get_order_book, token1)
        
        # Debug: Print the order book structure
        print(f"Order book structure: {dir(book)}")
        if hasattr(book, 'bids'):
            print(f"Bids structure: {book.bids[:2] if book.bids else 'Empty'}")
        if hasattr(book, 'asks'):
            print(f"Asks structure: {book.asks[:2] if book.asks else 'Empty'}")
    except Exception as e:
        print(f"Failed to get order book after retries: {e}")
        raise
    
    bids = pd.DataFrame()
    asks = pd.DataFrame()

    try:
        if hasattr(book, 'bids') and book.bids:
            # Convert OrderSummary objects to dictionaries with float values
            bids_data = [{'price': float(b.price), 'size': float(b.size)} for b in book.bids]
            bids = pd.DataFrame(bids_data)
            print(f"Bids DataFrame: {bids.head()}")
    except Exception as e:
        print(f"Error processing bids: {e}")
        pass

    try:
        if hasattr(book, 'asks') and book.asks:
            # Convert OrderSummary objects to dictionaries with float values
            asks_data = [{'price': float(a.price), 'size': float(a.size)} for a in book.asks]
            asks = pd.DataFrame(asks_data)
            print(f"Asks DataFrame: {asks.head()}")
    except Exception as e:
        print(f"Error processing asks: {e}")
        pass

    try:
        ret['best_bid'] = bids.iloc[-1]['price'] if not bids.empty else 0
    except Exception as e:
        print(f"Error getting best bid: {e}")
        ret['best_bid'] = 0

    try:
        ret['best_ask'] = asks.iloc[-1]['price'] if not asks.empty else 0
    except Exception as e:
        print(f"Error getting best ask: {e}")
        ret['best_ask'] = 0

    ret['midpoint'] = (ret['best_bid'] + ret['best_ask']) / 2
    
    if 'minimum_tick_size' not in row:
        print("Warning: 'minimum_tick_size' field missing, using default value")
        TICK_SIZE = 0.01
    else:
        TICK_SIZE = row['minimum_tick_size']
    
    ret['tick_size'] = TICK_SIZE

    bid_from, bid_to, ask_from, ask_to = get_bid_ask_range(ret, TICK_SIZE)
    v = round((ret['max_spread'] / 100), 2)

    bids_df = pd.DataFrame()
    bids_df['price'] = generate_numbers(bid_from, bid_to, TICK_SIZE)

    asks_df = pd.DataFrame()
    asks_df['price'] = generate_numbers(ask_from, ask_to, TICK_SIZE)

    try:
        bids_df = bids_df.merge(bids, on='price', how='left').fillna(0)
    except Exception as e:
        print(f"Error merging bids: {e}")
        bids_df = pd.DataFrame()

    try:
        asks_df = asks_df.merge(asks, on='price', how='left').fillna(0)
    except Exception as e:
        print(f"Error merging asks: {e}")
        asks_df = pd.DataFrame()

    best_bid_reward = 0
    ret_bid = pd.DataFrame()

    try:
        ret_bid = add_formula_params(bids_df, ret['midpoint'], v, rate)
        best_bid_reward = round(ret_bid['reward_per_100'].max(), 2)
    except Exception as e:
        print(f"Error calculating bid reward: {e}")
        pass

    best_ask_reward = 0
    ret_ask = pd.DataFrame()

    try:
        ret_ask = add_formula_params(asks_df, ret['midpoint'], v, rate)
        best_ask_reward = round(ret_ask['reward_per_100'].max(), 2)
    except Exception as e:
        print(f"Error calculating ask reward: {e}")
        pass

    ret['bid_reward_per_100'] = best_bid_reward
    ret['ask_reward_per_100'] = best_ask_reward

    ret['sm_reward_per_100'] = round((best_bid_reward + best_ask_reward) / 2, 2)
    ret['gm_reward_per_100'] = round((best_bid_reward * best_ask_reward) ** 0.5, 2)

    if 'end_date_iso' in row:
        ret['end_date_iso'] = row['end_date_iso']
    else:
        ret['end_date_iso'] = None
    
    ret['market_slug'] = row.get('market_slug', '')
    ret['token1'] = token1
    ret['token2'] = token2
    ret['condition_id'] = row.get('condition_id', '')

    return ret

# JSON file containing market slugs
MANUAL_MARKETS_FILE = 'manual_markets.json'
# Sheet name for manual targets
MANUAL_TARGETS_SHEET = 'Manual Targets'

def update_sheet(data, worksheet):
    """
    Update a worksheet with data, preserving existing data and formatting
    """
    all_values = worksheet.get_all_values()
    existing_num_rows = len(all_values)
    existing_num_cols = len(all_values[0]) if all_values else 0

    num_rows, num_cols = data.shape
    max_rows = max(num_rows, existing_num_rows)
    max_cols = max(num_cols, existing_num_cols)

    # Create a DataFrame with the maximum size and fill it with empty strings
    padded_data = pd.DataFrame('', index=range(max_rows), columns=range(max_cols))

    # Update the padded DataFrame with the original data and its columns
    padded_data.iloc[:num_rows, :num_cols] = data.values
    padded_data.columns = list(data.columns) + [''] * (max_cols - num_cols)

    # Update the sheet with the padded DataFrame, including column headers
    set_with_dataframe(worksheet, padded_data, include_index=False, include_column_header=True, resize=True)

def get_market_by_slug(client, slug):
    """
    Get market data by slug using the Gamma API and then the CLOB API
    """
    try:
        # Step 1: Call the Polymarket Gamma API to get the market data by slug
        url = f"https://gamma-api.polymarket.com/markets?slug={slug}"
        print(f"Making API request to: {url}")
        
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        markets_data = response.json()
        
        # Check if the response data is valid
        if not markets_data or not isinstance(markets_data, list) or len(markets_data) == 0:
            print(f"No market found with slug: {slug}")
            return None
        
        # Get the first market (should be the only one with this slug)
        market_data = markets_data[0]
        
        # Extract the condition_id
        condition_id = market_data.get('conditionId')
        if not condition_id:
            print(f"No condition ID found for market with slug: {slug}")
            return None
        
        print(f"Found condition ID for {slug}: {condition_id}")
        
        # Step 2: Use the condition_id to get the full market data from the CLOB API
        market = client.get_market(condition_id)
        
        # Debug: Print the structure of the market data
        print(f"Market data structure: {list(market.keys()) if market else 'None'}")
        
        # Add the market_slug to the market data for reference
        if market:
            market['market_slug'] = slug
        
        return market
    except Exception as e:
        print(f"Error fetching market with slug '{slug}': {e}")
        return None

def get_market_by_condition_id(client, condition_id):
    """
    Get market data directly by condition ID
    """
    try:
        print(f"Fetching market with condition ID: {condition_id}")
        
        # Use the condition_id to get the full market data from the CLOB API
        market = client.get_market(condition_id)
        
        # Debug: Print the structure of the market data
        print(f"Market data structure: {list(market.keys()) if market else 'None'}")
        
        return market
    except Exception as e:
        print(f"Error fetching market with condition ID '{condition_id}': {e}")
        return None

def fetch_and_process_manual_markets():
    """
    Fetch and process markets from the manual_markets.json file
    """
    # Initialize client and spreadsheet
    client = get_clob_client()
    spreadsheet = get_spreadsheet()
    
    # Create or get the Manual Targets worksheet
    try:
        wk_manual = spreadsheet.worksheet(MANUAL_TARGETS_SHEET)
    except:
        # Create the worksheet if it doesn't exist
        wk_manual = spreadsheet.add_worksheet(title=MANUAL_TARGETS_SHEET, rows=1000, cols=30)
        print(f"Created new '{MANUAL_TARGETS_SHEET}' worksheet")
    
    # Get the All Markets worksheet
    try:
        wk_all_markets = spreadsheet.worksheet('All Markets')
    except:
        print("Error: 'All Markets' worksheet not found. This is required for the system to work.")
        return
    
    # Load markets data from JSON file
    try:
        with open(MANUAL_MARKETS_FILE, 'r') as f:
            manual_markets_data = json.load(f)
        
        markets = manual_markets_data.get('markets', [])
        if not markets:
            print(f"No markets found in {MANUAL_MARKETS_FILE}")
            return
        
        print(f"Found {len(markets)} markets in {MANUAL_MARKETS_FILE}")
    except FileNotFoundError:
        print(f"Error: {MANUAL_MARKETS_FILE} not found. Please create this file with your market specifications.")
        return
    except json.JSONDecodeError:
        print(f"Error: {MANUAL_MARKETS_FILE} is not valid JSON. Please check the file format.")
        return
    
    # Fetch and process each market
    market_results = []
    
    for market_spec in markets:
        # Handle different formats of market specifications
        if isinstance(market_spec, str):
            # Legacy format: just a slug string
            slug = market_spec
            print(f"Processing market by slug: {slug}")
            market = get_market_by_slug(client, slug)
            market_id = slug  # For error reporting
        elif isinstance(market_spec, dict):
            if 'slug' in market_spec:
                # New format: object with slug
                slug = market_spec['slug']
                print(f"Processing market by slug: {slug}")
                market = get_market_by_slug(client, slug)
                market_id = slug  # For error reporting
            elif 'condition_id' in market_spec:
                # New format: object with condition_id
                condition_id = market_spec['condition_id']
                print(f"Processing market by condition ID: {condition_id}")
                market = get_market_by_condition_id(client, condition_id)
                market_id = condition_id  # For error reporting
            else:
                print(f"Invalid market specification: {market_spec}. Must contain 'slug' or 'condition_id'.")
                continue
        else:
            print(f"Invalid market specification format: {market_spec}. Must be a string or an object.")
            continue
        
        if market:
            try:
                result = process_single_row(market, client)
                if result:
                    market_results.append(result)
                    print(f"Successfully processed market: {market_id}")
            except Exception as e:
                print(f"Error processing market {market_id}: {e}")
    
    if not market_results:
        print("No markets were successfully processed.")
        return
    
    # Create DataFrame from results
    markets_df = pd.DataFrame(market_results)
    markets_df['spread'] = abs(markets_df['best_ask'] - markets_df['best_bid'])
    
    # Add volatility data with reduced workers to avoid rate limiting
    print("Calculating volatility data...")
    markets_df = add_volatility_to_df(markets_df, max_workers=2)
    markets_df['volatility_sum'] = markets_df['24_hour'] + markets_df['7_day'] + markets_df['14_day']
    
    markets_df = markets_df.sort_values('gm_reward_per_100', ascending=False)
    markets_df['volatilty/reward'] = ((markets_df['gm_reward_per_100'] / markets_df['volatility_sum']).round(2)).astype(str)
    
    # Select the same columns as for other markets
    markets_df = markets_df[['question', 'answer1', 'answer2', 'spread', 'rewards_daily_rate', 
                           'gm_reward_per_100', 'sm_reward_per_100', 'bid_reward_per_100', 'ask_reward_per_100',  
                           'volatility_sum', 'volatilty/reward', 'min_size', '1_hour', '3_hour', '6_hour', 
                           '12_hour', '24_hour', '7_day', '30_day', 'best_bid', 'best_ask', 'volatility_price', 
                           'max_spread', 'tick_size', 'neg_risk', 'market_slug', 'token1', 'token2', 'condition_id']]
    
    print(f"Processed {len(markets_df)} markets successfully.")
    
    # Update the Manual Targets worksheet
    update_sheet(markets_df, wk_manual)
    print(f"Updated '{MANUAL_TARGETS_SHEET}' worksheet.")
    
    # Also update the All Markets worksheet with the same data
    # First, get existing data from All Markets
    all_markets_data = pd.DataFrame(wk_all_markets.get_all_records())
    
    # Remove any existing entries for these markets to avoid duplicates
    if not all_markets_data.empty and 'question' in all_markets_data.columns:
        # Create a mask for rows that are NOT in our new markets_df
        mask = ~all_markets_data['question'].isin(markets_df['question'])
        # Keep only those rows
        all_markets_data = all_markets_data[mask]
    
    # Append our new market data
    if all_markets_data.empty:
        combined_df = markets_df
    else:
        # Ensure columns match before concatenating
        common_cols = list(set(all_markets_data.columns) & set(markets_df.columns))
        combined_df = pd.concat([all_markets_data[common_cols], markets_df[common_cols]], ignore_index=True)
    
    # Replace any infinite values with a large number to avoid JSON errors
    combined_df = combined_df.replace([np.inf, -np.inf], 999999)
    
    # Convert any NaN values to empty strings for Google Sheets
    combined_df = combined_df.fillna('')
    
    # Update the All Markets worksheet
    update_sheet(combined_df, wk_all_markets)
    print(f"Updated 'All Markets' worksheet with {len(markets_df)} markets.")

if __name__ == "__main__":
    try:
        fetch_and_process_manual_markets()
    except Exception as e:
        traceback.print_exc()
        print(str(e))
