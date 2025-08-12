import pandas as pd
import numpy as np
import json
import datetime
import os
import requests
from dotenv import load_dotenv
from data_updater.google_utils import get_spreadsheet
import poly_data.global_state as global_state
from data_updater.trading_utils import get_clob_client
from poly_data.polymarket_client import PolymarketClient
from poly_data.abis import erc20_abi
from web3 import Web3
from web3.middleware import geth_poa_middleware

# Load environment variables from .env file
load_dotenv()

# Constants
MANUAL_TARGETS_SHEET = 'Manual Targets'
OUTPUT_FILE = 'ai_analysis_prompt.txt'

# Parameter sets
PARAMETER_SETS = {
    'very': {
        'stop_loss_threshold': -1.25,
        'take_profit_threshold': 1.5,
        'spread_threshold': 0.05,
        'vol_window': 30,
        'sleep_period': 0,
        'volatility_threshold': 1000
    },
    'high': {
        'stop_loss_threshold': -1.25,
        'take_profit_threshold': 1.5,
        'spread_threshold': 0.05,
        'vol_window': 30,
        'sleep_period': 0,
        'volatility_threshold': 1000
    },
    'mid': {
        'stop_loss_threshold': -1.5,
        'take_profit_threshold': 1.5,
        'spread_threshold': 0.04,
        'vol_window': 20,
        'sleep_period': 2,
        'volatility_threshold': 1000
    },
    'shit': {
        'stop_loss_threshold': -3,
        'take_profit_threshold': 3,
        'spread_threshold': 0.05,
        'vol_window': 20,
        'sleep_period': 2,
        'volatility_threshold': 1000
    }
}

def get_manual_markets_data():
    """
    Fetch data from the Manual Targets worksheet
    """
    try:
        spreadsheet = get_spreadsheet()
        worksheet = spreadsheet.worksheet(MANUAL_TARGETS_SHEET)
        
        # Get all data from the worksheet
        data = worksheet.get_all_records()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            print(f"No data found in {MANUAL_TARGETS_SHEET} worksheet")
            return None
        
        print(f"Successfully fetched {len(df)} markets from {MANUAL_TARGETS_SHEET}")
        return df
    except Exception as e:
        print(f"Error fetching data from {MANUAL_TARGETS_SHEET}: {e}")
        return None

def analyze_market(market):
    """
    Analyze a single market and generate insights
    """
    analysis = {}
    
    # Basic market info
    analysis['question'] = market['question']
    analysis['answer1'] = market['answer1']
    analysis['answer2'] = market['answer2']
    
    # Convert best_bid to float for calculations
    try:
        best_bid = float(market['best_bid'])
    except (ValueError, TypeError):
        best_bid = 0.5  # Default to 50% if conversion fails
    
    analysis['current_prices'] = {
        market['answer1']: best_bid,
        market['answer2']: 1 - best_bid
    }
    analysis['spread'] = market['spread']
    
    # Volatility analysis
    volatility_metrics = {
        '1_hour': market['1_hour'],
        '3_hour': market['3_hour'],
        '6_hour': market['6_hour'],
        '12_hour': market['12_hour'],
        '24_hour': market['24_hour'],
        '7_day': market['7_day'],
        '30_day': market.get('30_day', 'N/A')
    }
    analysis['volatility'] = volatility_metrics
    analysis['volatility_sum'] = market['volatility_sum']
    
    # Reward analysis
    reward_metrics = {
        'rewards_daily_rate': market['rewards_daily_rate'],
        'gm_reward_per_100': market['gm_reward_per_100'],
        'sm_reward_per_100': market['sm_reward_per_100'],
        'bid_reward_per_100': market['bid_reward_per_100'],
        'ask_reward_per_100': market['ask_reward_per_100'],
        'volatility_reward_ratio': market['volatilty/reward']
    }
    analysis['rewards'] = reward_metrics
    
    # Market characteristics
    analysis['characteristics'] = {
        'min_size': market['min_size'],
        'max_spread': market['max_spread'],
        'tick_size': market['tick_size'],
        'neg_risk': market['neg_risk']
    }
    
    # Recommend parameter set
    analysis['recommended_params'] = recommend_parameters(market)
    
    return analysis

def recommend_parameters(market):
    """
    Recommend the best parameter set for a market based on its characteristics
    """
    # Convert values to float to ensure proper comparisons
    try:
        volatility = float(market['volatility_sum'])
    except (ValueError, TypeError):
        volatility = 0.0
        
    try:
        spread = float(market['spread'])
    except (ValueError, TypeError):
        spread = 0.0
        
    try:
        reward = float(market['gm_reward_per_100'])
    except (ValueError, TypeError):
        reward = 0.0
    
    # Logic for parameter recommendation
    if volatility < 5 and reward > 0.5:
        recommendation = 'very'
        reason = "Low volatility with good reward potential makes this market suitable for aggressive parameters."
    elif volatility < 10 and reward > 0.3:
        recommendation = 'high'
        reason = "Moderate volatility with decent reward potential suggests using high-tier parameters."
    elif volatility < 20:
        recommendation = 'mid'
        reason = "Medium volatility indicates a need for more balanced parameters."
    else:
        recommendation = 'shit'
        reason = "High volatility requires conservative parameters to manage risk."
    
    # Special cases
    if spread > 0.1:
        if recommendation != 'shit':
            recommendation = 'mid'
            reason += " However, the wide spread suggests using more moderate parameters."
    
    if reward < 0.1:
        recommendation = 'shit'
        reason = "Very low reward potential doesn't justify aggressive parameters."
    
    # Custom parameter suggestion
    custom_params = None
    custom_reason = None
    
    # If the market has unusual characteristics, suggest custom parameters
    if volatility > 30 and reward > 1.0:
        custom_params = {
            'stop_loss_threshold': -2.0,
            'take_profit_threshold': 2.5,
            'spread_threshold': 0.06,
            'vol_window': 15,
            'sleep_period': 1,
            'volatility_threshold': 1500
        }
        custom_reason = "This market has extremely high volatility but also high reward potential. Custom parameters are suggested to balance risk and reward."
    
    return {
        'recommended_type': recommendation,
        'parameters': PARAMETER_SETS[recommendation],
        'reason': reason,
        'custom_parameters': custom_params,
        'custom_reason': custom_reason
    }

def generate_overall_analysis(markets_data, analyses):
    """
    Generate an overall analysis of all markets
    """
    overall = {}
    
    # Convert numeric columns to float
    for col in ['volatility_sum', 'gm_reward_per_100', 'spread']:
        try:
            markets_data[col] = pd.to_numeric(markets_data[col], errors='coerce')
        except:
            markets_data[col] = 0.0
    
    # Basic stats
    overall['total_markets'] = len(analyses)
    overall['avg_volatility'] = markets_data['volatility_sum'].mean()
    overall['avg_reward'] = markets_data['gm_reward_per_100'].mean()
    overall['avg_spread'] = markets_data['spread'].mean()
    
    # Parameter recommendations summary
    param_counts = {}
    for analysis in analyses:
        param_type = analysis['recommended_params']['recommended_type']
        param_counts[param_type] = param_counts.get(param_type, 0) + 1
    
    overall['parameter_recommendations'] = param_counts
    
    # Best opportunities
    markets_data['reward_vol_ratio'] = markets_data['gm_reward_per_100'] / (markets_data['volatility_sum'] + 0.001)
    best_opportunities = markets_data.nlargest(3, 'reward_vol_ratio')
    overall['best_opportunities'] = best_opportunities['question'].tolist()
    
    return overall

def create_ai_prompt(overall_analysis, market_analyses):
    """
    Create a comprehensive AI prompt for analysis
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Handle NaN values in overall statistics
    avg_volatility = overall_analysis['avg_volatility']
    avg_reward = overall_analysis['avg_reward']
    avg_spread = overall_analysis['avg_spread']
    
    if pd.isna(avg_volatility):
        avg_volatility = 0.0
    if pd.isna(avg_reward):
        avg_reward = 0.0
    if pd.isna(avg_spread):
        avg_spread = 0.0
    
    # Get account balance information
    account_balance = overall_analysis.get('account_balance', {
        'usdc_balance': 0,
        'position_value': 0,
        'total_balance': 0,
        'position_percentage': 0
    })
    
    prompt = f"""# Polymarket Trading Analysis Request
Generated on: {now}

## Overview
I have data on {overall_analysis['total_markets']} markets from Polymarket that I'm considering for trading with my market making bot. I need your help to analyze these markets and provide insights on market making strategies and parameter selection.

## Market Making Strategy
This analysis is for a market making bot that provides liquidity on Polymarket prediction markets. The bot:
- Places buy and sell orders on both sides of the market to capture spreads
- Earns rewards for providing liquidity (shown as reward metrics in the data)
- Uses different parameter sets to adjust risk tolerance and trading aggressiveness
- Manages positions based on volatility, spread, and reward potential

The parameter sets (VERY, HIGH, MID, SHIT) control how the bot behaves:
- More aggressive sets (VERY, HIGH) use tighter stop-losses and take-profits, allowing for larger positions
- More conservative sets (MID, SHIT) use wider stop-losses and take-profits, with smaller positions
- The bot adjusts its behavior based on market volatility and reward metrics

### Account Overview
- USDC Balance: ${account_balance['usdc_balance']:.2f}
- Position Value: ${account_balance['position_value']:.4f}
- Total Account Value: ${account_balance['total_balance']:.2f}
- Percentage in Positions: {account_balance['position_percentage']:.1f}%

### Overall Market Statistics
- Total Markets: {overall_analysis['total_markets']}
- Average Volatility: {avg_volatility:.2f}
- Average Reward Potential: {avg_reward:.2f}
- Average Spread: {avg_spread:.4f}

### Parameter Type Distribution
"""

    # Add parameter distribution
    for param_type, count in overall_analysis['parameter_recommendations'].items():
        percentage = (count / overall_analysis['total_markets']) * 100
        prompt += f"- {param_type.upper()}: {count} markets ({percentage:.1f}%)\n"
    
    prompt += f"""
### Best Opportunities (Highest Reward-to-Volatility Ratio)
"""
    for opportunity in overall_analysis['best_opportunities']:
        prompt += f"- {opportunity}\n"
    
    prompt += f"""
## Parameter Sets
These are the standard parameter sets I use for different market conditions:

"""
    # Add parameter sets
    for param_type, params in PARAMETER_SETS.items():
        prompt += f"### {param_type.upper()}\n"
        for param_name, value in params.items():
            prompt += f"- {param_name}: {value}\n"
        prompt += "\n"
    
    prompt += f"""
## Detailed Market Analyses

I need your analysis on each market, including:
1. Assessment of the market's volatility and reward potential
2. Evaluation of the recommended parameter set
3. Any suggestions for custom parameters if appropriate
4. Trading strategy recommendations specific to each market

Here are the markets:

"""
    
    # Add individual market analyses
    for i, analysis in enumerate(market_analyses, 1):
        prompt += f"""### Market {i}: {analysis['question']}
- Current Prices: {analysis['answer1']}: {analysis['current_prices'][analysis['answer1']]:.3f}, {analysis['answer2']}: {analysis['current_prices'][analysis['answer2']]:.3f}
- Spread: {analysis['spread']:.4f}

**Volatility Metrics:**
- 1 Hour: {analysis['volatility']['1_hour']}
- 3 Hour: {analysis['volatility']['3_hour']}
- 6 Hour: {analysis['volatility']['6_hour']}
- 12 Hour: {analysis['volatility']['12_hour']}
- 24 Hour: {analysis['volatility']['24_hour']}
- 7 Day: {analysis['volatility']['7_day']}
- 30 Day: {analysis['volatility']['30_day']}
- Volatility Sum: {analysis['volatility_sum']}

**Reward Metrics:**
- Daily Rate: {analysis['rewards']['rewards_daily_rate'] if analysis['rewards']['rewards_daily_rate'] and not pd.isna(analysis['rewards']['rewards_daily_rate']) else 'N/A'}
- GM Reward per 100: {analysis['rewards']['gm_reward_per_100'] if analysis['rewards']['gm_reward_per_100'] and not pd.isna(analysis['rewards']['gm_reward_per_100']) else 'N/A'}
- SM Reward per 100: {analysis['rewards']['sm_reward_per_100'] if analysis['rewards']['sm_reward_per_100'] and not pd.isna(analysis['rewards']['sm_reward_per_100']) else 'N/A'}
- Bid Reward per 100: {analysis['rewards']['bid_reward_per_100'] if analysis['rewards']['bid_reward_per_100'] and not pd.isna(analysis['rewards']['bid_reward_per_100']) else 'N/A'}
- Ask Reward per 100: {analysis['rewards']['ask_reward_per_100'] if analysis['rewards']['ask_reward_per_100'] and not pd.isna(analysis['rewards']['ask_reward_per_100']) else 'N/A'}
- Volatility/Reward Ratio: {analysis['rewards']['volatility_reward_ratio'] if analysis['rewards']['volatility_reward_ratio'] and not pd.isna(analysis['rewards']['volatility_reward_ratio']) else 'N/A'}

**Market Characteristics:**
- Min Size: {analysis['characteristics']['min_size']}
- Max Spread: {analysis['characteristics']['max_spread']}
- Tick Size: {analysis['characteristics']['tick_size']}
- Negative Risk: {analysis['characteristics']['neg_risk']}

**Recommended Parameters:** {analysis['recommended_params']['recommended_type'].upper()}
- Reason: {analysis['recommended_params']['reason']}
"""
        
        # Add custom parameters if available
        if analysis['recommended_params']['custom_parameters']:
            prompt += f"""
**Custom Parameter Suggestion:**
- Reason: {analysis['recommended_params']['custom_reason']}
"""
            for param_name, value in analysis['recommended_params']['custom_parameters'].items():
                prompt += f"- {param_name}: {value}\n"
        
        prompt += "\n\n"
    
    prompt += f"""
## Questions for Analysis

Based on the data provided for my market making bot:

1. For each market, do you agree with the recommended parameter set? If not, which set would you recommend and why?

2. Are there any markets where you would suggest completely custom parameters for market making? Please provide specific values and reasoning.

3. Which markets present the best market making opportunities right now, and what specific bid/ask strategies would you recommend?

4. How should I manage my spread in each market to balance between earning rewards and getting filled?

5. Are there any markets that should be avoided for market making? Please explain your reasoning.

6. Given my current account balance (${account_balance['total_balance']:.2f}) with ${account_balance['usdc_balance']:.2f} in USDC and ${account_balance['position_value']:.4f} in positions, what position sizing would you recommend for each market? Consider both absolute dollar amounts and percentage of portfolio.

7. For the top 3 opportunities, please provide a detailed market making plan including:
   - Optimal bid/ask placement strategy
   - Position sizing and inventory management
   - Risk management approach (stop-loss and take-profit levels)
   - Rebalancing strategy when positions become uneven
   - Monitoring recommendations

8. Based on my current account allocation ({account_balance['position_percentage']:.1f}% in positions), would you recommend increasing or decreasing market making exposure? Why?

9. How should I adjust my market making parameters during periods of high volatility versus low volatility?

10. Are there any patterns or insights across these markets that could inform a more effective market making strategy?

Thank you for your analysis!
"""
    
    return prompt

def get_account_balance():
    """
    Get account balance information from the Polymarket client
    """
    try:
        # Get environment variables
        browser_address = os.getenv("BROWSER_ADDRESS")
        if not browser_address:
            print("BROWSER_ADDRESS environment variable not set")
            return {
                'usdc_balance': 0,
                'position_value': 0,
                'total_balance': 0,
                'position_percentage': 0
            }
        
        # Initialize Web3 connection to Polygon
        web3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
        web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Convert address to checksum format using the correct method
        # In newer versions of Web3.py, use web3.to_checksum_address instead of Web3.toChecksumAddress
        try:
            if hasattr(web3, 'to_checksum_address'):
                checksum_address = web3.to_checksum_address(browser_address)
            elif hasattr(web3, 'toChecksumAddress'):
                checksum_address = web3.toChecksumAddress(browser_address)
            else:
                # Try importing from eth_utils as a fallback
                from eth_utils import to_checksum_address
                checksum_address = to_checksum_address(browser_address)
        except Exception as e:
            print(f"Error converting address to checksum format: {e}")
            # Use the address as-is if conversion fails
            checksum_address = browser_address
        
        # Set up USDC contract for balance checks
        usdc_contract = web3.eth.contract(
            address="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", 
            abi=erc20_abi
        )
        
        # Get USDC balance
        usdc_balance = usdc_contract.functions.balanceOf(checksum_address).call() / 10**6
        
        # Get position value from API
        res = requests.get(f'https://data-api.polymarket.com/value?user={checksum_address}')
        res_json = res.json()
        print(f"API response: {res_json}")
        
        # Handle different response formats
        if isinstance(res_json, dict) and 'value' in res_json:
            position_value = float(res_json['value'])
        elif isinstance(res_json, list) and len(res_json) > 0 and 'value' in res_json[0]:
            position_value = float(res_json[0]['value'])
        else:
            print(f"Unexpected API response format: {res_json}")
            position_value = 0.0
        
        # Calculate total balance
        total_balance = usdc_balance + position_value
        
        # Calculate percentage in positions
        if total_balance > 0:
            position_percentage = (position_value / total_balance) * 100
        else:
            position_percentage = 0
            
        return {
            'usdc_balance': usdc_balance,
            'position_value': position_value,
            'total_balance': total_balance,
            'position_percentage': position_percentage
        }
    except Exception as e:
        print(f"Error getting account balance: {e}")
        return {
            'usdc_balance': 0,
            'position_value': 0,
            'total_balance': 0,
            'position_percentage': 0
        }

def main():
    # Get market data
    markets_data = get_manual_markets_data()
    if markets_data is None:
        return
    
    print(f"Analyzing {len(markets_data)} markets...")
    
    # Get account balance information
    print("Fetching account balance information...")
    account_balance = get_account_balance()
    
    # Analyze each market
    market_analyses = []
    for _, market in markets_data.iterrows():
        analysis = analyze_market(market)
        market_analyses.append(analysis)
    
    # Generate overall analysis
    overall_analysis = generate_overall_analysis(markets_data, market_analyses)
    
    # Add account balance to overall analysis
    overall_analysis['account_balance'] = account_balance
    
    # Create AI prompt
    prompt = create_ai_prompt(overall_analysis, market_analyses)
    
    # Save prompt to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(prompt)
    
    print(f"Analysis complete! AI prompt saved to {OUTPUT_FILE}")
    print(f"Copy the contents of {OUTPUT_FILE} and paste it to your preferred AI service for detailed analysis.")

if __name__ == "__main__":
    main()
