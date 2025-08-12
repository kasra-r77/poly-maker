import time
import pandas as pd
from data_updater.trading_utils import get_clob_client
from data_updater.google_utils import get_spreadsheet
from data_updater.find_markets import get_sel_df, get_all_markets, get_all_results, get_markets, add_volatility_to_df, process_single_row, filter_crypto_markets
from gspread_dataframe import set_with_dataframe
import traceback

# Initialize global variables
spreadsheet = get_spreadsheet()
client = get_clob_client()

wk_all = spreadsheet.worksheet("All Markets")
wk_vol = spreadsheet.worksheet("Volatility Markets")
wk_crypto = None
try:
    wk_crypto = spreadsheet.worksheet("Crypto Markets")
except:
    # Worksheet doesn't exist yet, will be created later
    pass

sel_df = get_sel_df(spreadsheet, "Selected Markets")

def update_sheet(data, worksheet):
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

def sort_df(df):
    # Calculate the mean and standard deviation for each column
    mean_gm = df['gm_reward_per_100'].mean()
    std_gm = df['gm_reward_per_100'].std()
    
    mean_volatility = df['volatility_sum'].mean()
    std_volatility = df['volatility_sum'].std()
    
    # Standardize the columns
    df['std_gm_reward_per_100'] = (df['gm_reward_per_100'] - mean_gm) / std_gm
    df['std_volatility_sum'] = (df['volatility_sum'] - mean_volatility) / std_volatility
    
    # Define a custom scoring function for best_bid and best_ask
    def proximity_score(value):
        if 0.1 <= value <= 0.25:
            return (0.25 - value) / 0.15
        elif 0.75 <= value <= 0.9:
            return (value - 0.75) / 0.15
        else:
            return 0
    
    df['bid_score'] = df['best_bid'].apply(proximity_score)
    df['ask_score'] = df['best_ask'].apply(proximity_score)
    
    # Create a composite score (higher is better for rewards, lower is better for volatility, with proximity scores)
    df['composite_score'] = (
        df['std_gm_reward_per_100'] - 
        df['std_volatility_sum'] + 
        df['bid_score'] + 
        df['ask_score']
    )
    
    # Sort by the composite score in descending order
    sorted_df = df.sort_values(by='composite_score', ascending=False)
    
    # Drop the intermediate columns used for calculation
    sorted_df = sorted_df.drop(columns=['std_gm_reward_per_100', 'std_volatility_sum', 'bid_score', 'ask_score', 'composite_score'])
    
    return sorted_df

def fetch_and_process_data():
    global spreadsheet, client, wk_all, wk_vol, wk_crypto, sel_df
    
    spreadsheet = get_spreadsheet()
    client = get_clob_client()

    wk_all = spreadsheet.worksheet("All Markets")
    wk_vol = spreadsheet.worksheet("Volatility Markets")
    wk_full = spreadsheet.worksheet("Full Markets")
    
    # Create or get the Crypto Markets worksheet
    try:
        wk_crypto = spreadsheet.worksheet("Crypto Markets")
    except:
        # Create the worksheet if it doesn't exist
        wk_crypto = spreadsheet.add_worksheet(title="Crypto Markets", rows=1000, cols=30)
        print("Created new 'Crypto Markets' worksheet")

    sel_df = get_sel_df(spreadsheet, "Selected Markets")

    all_df = get_all_markets(client)
    print("Got all Markets")
    
    # Filter crypto markets
    crypto_df = filter_crypto_markets(all_df)
    print(f"Found {len(crypto_df)} crypto markets")
    
    # Reduce the number of concurrent workers to avoid rate limiting
    all_results = get_all_results(all_df, client, max_workers=3)
    print("Got all Results")
    m_data, all_markets = get_markets(all_results, sel_df, maker_reward=0.75)
    print("Got all orderbook")

    print(f'{pd.to_datetime("now")}: Fetched all markets data of length {len(all_markets)}.')
    # Reduce the number of concurrent workers for volatility data to avoid rate limiting
    new_df = add_volatility_to_df(all_markets, max_workers=2)
    new_df['volatility_sum'] =  new_df['24_hour'] + new_df['7_day'] + new_df['14_day']
    
    new_df = new_df.sort_values('volatility_sum', ascending=True)
    new_df['volatilty/reward'] = ((new_df['gm_reward_per_100'] / new_df['volatility_sum']).round(2)).astype(str)

    new_df = new_df[['question', 'answer1', 'answer2', 'spread', 'rewards_daily_rate', 'gm_reward_per_100', 'sm_reward_per_100', 'bid_reward_per_100', 'ask_reward_per_100',  'volatility_sum', 'volatilty/reward', 'min_size', '1_hour', '3_hour', '6_hour', '12_hour', '24_hour', '7_day', '30_day',  
                     'best_bid', 'best_ask', 'volatility_price', 'max_spread', 'tick_size',  
                     'neg_risk',  'market_slug', 'token1', 'token2', 'condition_id']]

    
    volatility_df = new_df.copy()
    volatility_df = volatility_df[new_df['volatility_sum'] < 20]
    # volatility_df = sort_df(volatility_df)
    volatility_df = volatility_df.sort_values('gm_reward_per_100', ascending=False)
   
    new_df = new_df.sort_values('gm_reward_per_100', ascending=False)
    
    # Process crypto markets if any were found
    if len(crypto_df) > 0:
        # Get results for crypto markets
        crypto_results = []
        crypto_error_log = []
        
        for idx, row in crypto_df.iterrows():
            try:
                result = process_single_row(row, client)
                if result:
                    crypto_results.append(result)
            except Exception as e:
                error_info = {
                    'index': idx,
                    'question': row.get('question', 'Unknown'),
                    'market_slug': row.get('market_slug', 'Unknown'),
                    'error': str(e)
                }
                crypto_error_log.append(error_info)
                print(f"Error processing crypto market: {error_info['question']} - {error_info['error']}")
        
        # Save crypto market errors to file
        if crypto_error_log:
            crypto_error_df = pd.DataFrame(crypto_error_log)
            crypto_error_df.to_csv('crypto_market_errors.csv', index=False)
            print(f"Saved {len(crypto_error_log)} crypto market errors to crypto_market_errors.csv")
        
        if crypto_results:
            crypto_markets_df = pd.DataFrame(crypto_results)
            crypto_markets_df['spread'] = abs(crypto_markets_df['best_ask'] - crypto_markets_df['best_bid'])
            
            # Add volatility data with reduced workers to avoid rate limiting
            crypto_markets_df = add_volatility_to_df(crypto_markets_df, max_workers=2)
            crypto_markets_df['volatility_sum'] = crypto_markets_df['24_hour'] + crypto_markets_df['7_day'] + crypto_markets_df['14_day']
            
            crypto_markets_df = crypto_markets_df.sort_values('gm_reward_per_100', ascending=False)
            crypto_markets_df['volatilty/reward'] = ((crypto_markets_df['gm_reward_per_100'] / crypto_markets_df['volatility_sum']).round(2)).astype(str)
            
            # Select the same columns as for other markets
            crypto_markets_df = crypto_markets_df[['question', 'answer1', 'answer2', 'spread', 'rewards_daily_rate', 'gm_reward_per_100', 'sm_reward_per_100', 'bid_reward_per_100', 'ask_reward_per_100',  'volatility_sum', 'volatilty/reward', 'min_size', '1_hour', '3_hour', '6_hour', '12_hour', '24_hour', '7_day', '30_day',  
                         'best_bid', 'best_ask', 'volatility_price', 'max_spread', 'tick_size',  
                         'neg_risk',  'market_slug', 'token1', 'token2', 'condition_id']]
            
            print(f'{pd.to_datetime("now")}: Processed {len(crypto_markets_df)} crypto markets.')
            
            # Update the Crypto Markets worksheet
            update_sheet(crypto_markets_df, wk_crypto)
            print(f'{pd.to_datetime("now")}: Updated Crypto Markets worksheet.')

    print(f'{pd.to_datetime("now")}: Fetched select market of length {len(new_df)}.')

    if len(new_df) > 50:
        update_sheet(new_df, wk_all)
        update_sheet(volatility_df, wk_vol)
        update_sheet(m_data, wk_full)
    else:
        print(f'{pd.to_datetime("now")}: Not updating sheet because of length {len(new_df)}.')

if __name__ == "__main__":
    while True:
        try:
            fetch_and_process_data()
            time.sleep(60 * 60)  # Sleep for an hour
        except Exception as e:
            traceback.print_exc()
            print(str(e))
