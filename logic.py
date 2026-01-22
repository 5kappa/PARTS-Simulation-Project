import pandas as pd
import numpy as np
import json
import os

RIGS_FILE = 'data/saved_rigs.json'

def load_data():
    """Loads components and events data from CSV files."""
    try:
        # Load Components
        components_df = pd.read_csv('data/components.csv')
        
        # Standardize Columns for Components
        col_map_comps = {
            'name': 'Name', 'category': 'Category', 
            'price': 'BasePrice', 'volatility': 'Volatility'
        }
        components_df.rename(columns=col_map_comps, inplace=True)
        
        # Fallback capitalization if map didn't catch everything
        if 'BasePrice' not in components_df.columns:
             components_df.columns = [c.title() for c in components_df.columns]
             if 'Price' in components_df.columns: components_df.rename(columns={'Price': 'BasePrice'}, inplace=True)
        
        # Clean Price Column
        if components_df['BasePrice'].dtype == 'object':
            components_df['BasePrice'] = components_df['BasePrice'].astype(str).str.replace('$', '').str.replace(',', '').str.replace('+', '').str.strip()
            components_df['BasePrice'] = pd.to_numeric(components_df['BasePrice'], errors='coerce')
        
        components_df.dropna(subset=['BasePrice', 'Volatility'], inplace=True)

        # Load Events
        events_df = pd.read_csv('data/events.csv')
        
        # Standardize Columns for Events
        # Expected: EventName, Probability, Multiplier, target_type, target_detail
        # CSV has: event_name, probability, multiplier, target_type, target_detail
        col_map_events = {
            'event_name': 'EventName',
            'probability': 'Probability', 
            'multiplier': 'Multiplier'
        }
        events_df.rename(columns=col_map_events, inplace=True)
        
        return components_df, events_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_event_impact_matrix(components_df, events_df):
    """
    Pre-calculates the multiplier for every (Event, Component) pair.
    Returns a matrix of shape (n_events, n_components).
    """
    n_events = len(events_df)
    n_components = len(components_df)
    
    # Initialize with 1.0 (no impact)
    impact_matrix = np.ones((n_events, n_components))
    
    # Iterate over events to fill the matrix
    for i, event in events_df.iterrows():
        # 1. Category Filter
        if event['target_type'] == "ALL":
            mask = np.ones(n_components, dtype=bool)
        else:
            mask = (components_df['Category'] == event['target_type']).values

        # 2. Detail Filter
        detail = str(event['target_detail'])
        clean_detail = detail.replace('$', '').replace(',', '').strip()
        is_numeric_target = clean_detail.replace('.', '', 1).isdigit() if clean_detail else False
        
        if is_numeric_target:
            target_score = float(clean_detail)
            vol_scores = components_df['Volatility'].values
            score_mask = np.isclose(vol_scores, target_score, atol=0.01)
            mask = mask & score_mask
        elif detail != "ALL" and detail != "nan":
            name_mask = components_df['Name'].str.contains(detail, case=False, na=False).values
            mask = mask & name_mask

        # 3. Apply Multiplier (Using TitleCase 'Multiplier')
        impact_matrix[i, mask] = event['Multiplier']
        
    return impact_matrix

def run_simulation(selected_parts_indices, n_iterations=1000, components_df=None, events_df=None, time_horizon_months=12):
    """
    Runs a Monte Carlo simulation.
    time_horizon_months: Scaling factor for probability (default 12 months).
    P_t = 1 - (1 - P_annual)^(t/12)
    """
    if components_df is None or events_df is None:
        components_df, events_df = load_data()
        
    if not selected_parts_indices or components_df.empty or events_df.empty:
        return {
            'final_costs': [],
            'base_price': 0, 'mean_cost': 0, 'median_cost': 0,
            'worst_case': 0, 'best_case': 0
        }

    # Extract selected parts
    # selected_parts_indices is a list of index labels. We need integer positions (iloc).
    # Get integer locations of the selected indices
    selected_positions = [components_df.index.get_loc(idx) for idx in selected_parts_indices]
    
    # Calculate Impact Matrix (n_events, n_components)
    impact_matrix = calculate_event_impact_matrix(components_df, events_df)
    
    # Shape: (n_events, n_selected_parts)
    selected_impact_matrix = impact_matrix[:, selected_positions]
    
    # Base Prices of selected parts
    base_prices = components_df.loc[selected_parts_indices, 'BasePrice'].values
    total_base_price = base_prices.sum()
    
    # Simulation Logic
    num_events = len(events_df)
    
    # Scale Probabilities based on Time Horizon
    # Base probabilities in CSV are assumed Annual (12 months)
    annual_probs = events_df['Probability'].values
    
    if time_horizon_months == 12:
        event_probs = annual_probs
    else:
        # Scale: P_t = 1 - (1 - P)^t_ratio
        t_ratio = time_horizon_months / 12.0
        event_probs = 1 - (1 - annual_probs) ** t_ratio
    
    # 1. Generate Triggered Events (n_iterations, n_events) [0 or 1]
    random_matrix = np.random.rand(n_iterations, num_events)
    triggered_events = (random_matrix < event_probs).astype(float)
    
    # 2. Calculate Final Multipliers for each component in each iteration
    # We want: Component_Final_Multiplier = Product(triggered_event_multipliers)
    # Log Trick: Log(Final) = Sum( Triggered * Log(Event_Mult) )
    # Matrix Mult: (n_iterations, n_events) @ (n_events, n_selected_parts) 
    # Result: (n_iterations, n_selected_parts)
    
    log_impacts = np.log(selected_impact_matrix)
    log_final_multipliers = triggered_events @ log_impacts
    
    final_multipliers = np.exp(log_final_multipliers)
    
    # 3. Calculate Item Costs
    # Item_Cost[iter, item] = Base_Price[item] * Final_Multiplier[iter, item]
    # base_prices shape (n_selected,) broadcast against (n_iterations, n_selected)
    final_item_costs = final_multipliers * base_prices
    
    # 4. Total Build Cost per Iteration
    # Sum across items (axis 1)
    final_build_costs = final_item_costs.sum(axis=1)
    
    return {
        'final_costs': final_build_costs,
        'base_price': total_base_price,
        'mean_cost': np.mean(final_build_costs),
        'median_cost': np.median(final_build_costs),
        'cost_90th': np.percentile(final_build_costs, 90),
        'var_95': np.percentile(final_build_costs, 95),
        'worst_case': np.max(final_build_costs),
        'best_case': np.min(final_build_costs),
        'triggered_events': triggered_events  # Added for Sensitivity Analysis
    }

def calculate_sensitivity(simulation_results, events_df):
    """
    Calculates the 'Dollar Impact' of each event.
    Impact = Mean(Cost | Event Happened) - Mean(Cost | Event Didn't Happen)
    """
    triggered_events = simulation_results.get('triggered_events')
    final_costs = simulation_results.get('final_costs')
    
    if triggered_events is None or final_costs is None:
        return pd.DataFrame()
        
    impacts = []
    
    # Iterate through each event column index
    for i, event_row in events_df.iterrows():
        event_name = event_row['EventName']
        
        # Boolean mask: True where event i happened
        # triggered_events is (n_iterations, n_events)
        event_happened_mask = triggered_events[:, i] == 1.0
        
        # Only calculate if we have a mix of true/false (otherwise variance is 0)
        count_true = np.sum(event_happened_mask)
        count_total = len(final_costs)
        
        if count_true > 0 and count_true < count_total:
            cost_with = np.mean(final_costs[event_happened_mask])
            cost_without = np.mean(final_costs[~event_happened_mask])
            impact = cost_with - cost_without
            occurrence_rate = count_true / count_total
        else:
            impact = 0.0
            occurrence_rate = 1.0 if count_true == count_total else 0.0
            
        impacts.append({
            'Event': event_name, 
            'Impact ($)': impact,
            'Frequency': occurrence_rate
        })
        
    # Return sorted by absolute impact
    df = pd.DataFrame(impacts)
    if not df.empty:
        df = df.sort_values(by='Impact ($)', key=abs, ascending=False)
    return df

def load_builds():
    """Loads saved builds from JSON file."""
    if not os.path.exists(RIGS_FILE):
        return {}
    try:
        with open(RIGS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading rigs: {e}")
        return {}

def save_build(name, selected_indices):
    """Saves a build to the JSON file."""
    rigs = load_builds()
    # Convert to standard Python ints for JSON serialization
    clean_indices = [int(i) for i in selected_indices]
    rigs[name] = clean_indices
    try:
        with open(RIGS_FILE, 'w') as f:
            json.dump(rigs, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving rig: {e}")
        return False

def delete_build(name):
    """Deletes a build from the JSON file."""
    rigs = load_builds()
    if name in rigs:
        del rigs[name]
        try:
            with open(RIGS_FILE, 'w') as f:
                json.dump(rigs, f, indent=4)
            return True
        except Exception as e:
            print(f"Error deleting rig: {e}")
            return False
    return False
