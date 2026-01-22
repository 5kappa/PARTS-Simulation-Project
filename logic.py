import pandas as pd
import numpy as np

def load_data():
    """Loads components and events data from CSV files."""
    try:
        # Load Components
        components_df = pd.read_csv('data/components.csv')
        
        # Standardize Columns if user uploads different format
        # Map: name->Name, category->Category, price->BasePrice, volatility->Volatility
        col_map = {
            'name': 'Name', 'category': 'Category', 
            'price': 'BasePrice', 'volatility': 'Volatility'
        }
        components_df.rename(columns=col_map, inplace=True)
        
        # Ensure we have the required columns
        required_cols = ['Name', 'Category', 'BasePrice', 'Volatility']
        if not all(col in components_df.columns for col in required_cols):
             # Try to match case-insensitive
             components_df.columns = [c.title() for c in components_df.columns]
             # Remap 'Price' to 'BasePrice' if needed
             if 'Price' in components_df.columns and 'BasePrice' not in components_df.columns:
                 components_df.rename(columns={'Price': 'BasePrice'}, inplace=True)

        # Clean Price Column (remove '$' and ',')
        if components_df['BasePrice'].dtype == 'object':
            components_df['BasePrice'] = components_df['BasePrice'].astype(str).str.replace('$', '').str.replace(',', '').str.replace('+', '').str.strip()
            components_df['BasePrice'] = pd.to_numeric(components_df['BasePrice'], errors='coerce')
        
        # Drop rows with invalid data
        components_df.dropna(subset=['BasePrice', 'Volatility'], inplace=True)

        # Load Events
        events_df = pd.read_csv('data/events.csv')
        return components_df, events_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_build_base_price(selected_parts_indices, components_df):
    """Calculates the base price of the selected build."""
    if not selected_parts_indices:
        return 0.0
    
    # helper to process inputs as a list of indices or rows
    selected_parts = components_df.loc[selected_parts_indices]
    return selected_parts['BasePrice'].sum()

def run_simulation(selected_parts_indices, n_iterations=1000, components_df=None, events_df=None):
    """
    Runs a Monte Carlo simulation for the selected build.
    
    Args:
        selected_parts_indices (list): List of indices of the selected parts in components_df.
        n_iterations (int): Number of simulations to run.
        components_df (pd.DataFrame): Dataframe of all components.
        events_df (pd.DataFrame): Dataframe of all possible events.
        
    Returns:
        dict: Simulation results including 'final_costs', 'base_price', 'survival_rate'.
    """
    if components_df is None or events_df is None:
        components_df, events_df = load_data()
        
    if not selected_parts_indices or components_df.empty or events_df.empty:
        return {
            'final_costs': [],
            'base_price': 0,
            'mean_cost': 0,
            'median_cost': 0,
            'worst_case': 0,
            'best_case': 0
        }

    # Extract selected parts
    build_parts = components_df.loc[selected_parts_indices].copy()
    base_price = build_parts['BasePrice'].sum()
    
    # Pre-calculate probabilities and multipliers for vectorization
    event_probs = events_df['Probability'].values
    event_multipliers = events_df['Multiplier'].values
    num_events = len(events_df)
    
    # Simulation Logic
    # We want to simulate N iterations. 
    # In each iteration, each event might occur based on its probability.
    # If multiple events occur, we assume their multipliers compound (or sum? implementation choice).
    # Plan: Compounding multipliers makes sense for "shocks".
    # However, to avoid extreme explosions, let's just take the product of all triggered multipliers.
    # ALSO, we need to apply the specialized logic: 
    # "Price multipliers are applied... using part volatility as a dampener/amplifier?"
    # Implementation: 
    # Effective Multiplier for Part P = 1 + (Global_Multiplier - 1) * Volatility_of_P
    # So if Global Multiplier is 1.5 (50% increase) and Part Volatility is 0.5,
    # The part price increases by 0.5 * 50% = 25%.
    
    simulation_results = []
    
    # Optimized Vectorized approach
    # Shape: (n_iterations, num_events) -> boolean matrix of triggered events
    random_matrix = np.random.rand(n_iterations, num_events)
    triggered_events = random_matrix < event_probs
    
    # Calculate Global Multiplier for each iteration
    # Start with 1.0. For each triggered event, multiply by its multiplier.
    # We can use np.prod where triggered.
    
    # Create a matrix of multipliers. If not triggered, multiplier is 1.0.
    # where(condition, x, y) -> if triggered, use multiplier, else 1.0
    run_multipliers = np.where(triggered_events, event_multipliers, 1.0)
    
    # Product across events for each iteration -> (n_iterations,)
    global_multipliers = np.prod(run_multipliers, axis=1)
    
    # Now simulate part prices for each iteration
    # build_parts has 'BasePrice' and 'Volatility'
    # Cost_i = Sum( Part_j_Base * (1 + (Global_Multiplier_i - 1) * Part_j_Volatility) )
    
    # Let's vectorize this calculation too.
    # global_multipliers is array of size N
    # parts_base is array of size P
    # parts_volatility is array of size P
    
    # We need Sum over j of: Part_j_Base + Part_j_Base * Part_j_Volatility * (Global_Multiplier_i - 1)
    # = Sum(Part_j_Base) + (Global_Multiplier_i - 1) * Sum(Part_j_Base * Part_j_Volatility)
    
    total_base_price = build_parts['BasePrice'].sum()
    weighted_volatility_component = (build_parts['BasePrice'] * build_parts['Volatility']).sum()
    
    # Formula: Total_Cost_i = Base_Total + (Global_Mult_i - 1) * Weighted_Vol_Sum
    # Check logic: 
    # if Global_Mult is 1.0, cost = Base_Total. Correct.
    # if Global_Mult is 1.5, Cost = Base_Total + 0.5 * Sum(Price * Volatility). Correct.
    
    final_costs = total_base_price + (global_multipliers - 1) * weighted_volatility_component
    
    return {
        'final_costs': final_costs,
        'base_price': total_base_price,
        'mean_cost': np.mean(final_costs),
        'median_cost': np.median(final_costs),
        'worst_case': np.max(final_costs),
        'best_case': np.min(final_costs)
    }
