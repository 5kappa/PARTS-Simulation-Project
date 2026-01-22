import pandas as pd
import numpy as np

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

def run_simulation(selected_parts_indices, n_iterations=1000, components_df=None, events_df=None):
    """
    Runs a Monte Carlo simulation using the High-Precision Targeting Matrix.
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
    # We need to map selected_indices (which are likely DataFrame indices) to positions in the component_df
    # But wait, run_simulation receives selected_parts_indices which might be from the User Selection.
    # To keep the matrix aligned, we should perform calculations on ALL components, then just pick the selected ones.
    # OR, filter components first? 
    # Better: Pre-calculate the FULL matrix for all components (since n_components is small ~1000).
    
    # Calculate Impact Matrix (n_events, n_components)
    impact_matrix = calculate_event_impact_matrix(components_df, events_df)
    
    # Filter the matrix for only the *selected* components to save memory/time during simulation
    # selected_parts_indices is a list of index labels. We need integer positions (iloc).
    # Get integer locations of the selected indices
    selected_positions = [components_df.index.get_loc(idx) for idx in selected_parts_indices]
    
    # Shape: (n_events, n_selected_parts)
    selected_impact_matrix = impact_matrix[:, selected_positions]
    
    # Base Prices of selected parts
    base_prices = components_df.loc[selected_parts_indices, 'BasePrice'].values
    total_base_price = base_prices.sum()
    
    # Simulation Logic
    num_events = len(events_df)
    event_probs = events_df['Probability'].values
    
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
        'worst_case': np.max(final_build_costs),
        'best_case': np.min(final_build_costs)
    }
