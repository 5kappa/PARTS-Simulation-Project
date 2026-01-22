import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logic
import sys
from streamlit.web import cli as stcli

# Page Config (Must be the first Streamlit command)
st.set_page_config(page_title="Budget Resilience Simulator", layout="wide")

if __name__ == '__main__':
    if st.runtime.exists():
        # Code continues execution inside streamlit
        pass
    else:
        # Relaunch the script with streamlit
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

# Load Data
components_df, events_df = logic.load_data()

st.title("🖥️ PC Build Budget Resilience Simulator")
st.markdown("""
This tool uses **Monte Carlo Simulation** to test if your PC build budget can survive future market volatility.
It simulates **1,000+ market scenarios** (AI shortages, crypto booms, supply gluts) to estimate probabilistic costs.
""")

# Sidebar
st.sidebar.header("Simulation Settings")
budget_limit = st.sidebar.number_input("Max Budget ($)", min_value=500, max_value=10000, value=2000, step=100)
n_iterations = st.sidebar.slider("Simulation Iterations", min_value=100, max_value=10000, value=2000, step=100)

if not events_df.empty:
    st.sidebar.header("Market Events Preview")
    st.sidebar.dataframe(events_df[['EventName', 'Probability', 'Multiplier']], hide_index=True)
else:
    st.error("Events data could not be loaded!")

# Main UI - Component Selection
st.header("1. Select Your Components")

if components_df.empty:
    st.error("Components data could not be loaded! Please check components.csv.")
else:
    col1, col2, col3 = st.columns(3)

    # Helper to create selectboxes
    def create_selectbox(column, category_name):
        filtered = components_df[components_df['Category'] == category_name]
        if filtered.empty:
            column.warning(f"No parts found for {category_name}")
            return None
        
        # Create a nice label with price
        options = filtered.index
        # Map index to a label "Name ($Price) [Vol: 0.x]"
        labels = {i: f"{filtered.loc[i, 'Name']} (${filtered.loc[i, 'BasePrice']}) [Vol: {filtered.loc[i, 'Volatility']}]" for i in options}
        
        selected_idx = column.selectbox(
            f"Select {category_name}", 
            options=options, 
            format_func=lambda x: labels[x]
        )
        return selected_idx

    selected_indices = []
    # Standard Categories to look for
    categories = ['CPU', 'GPU', 'RAM', 'SSD', 'PSU', 'Case']
    # Use actual categories from file if they match roughly, or just fallback to unique categories
    available_cats = components_df['Category'].unique()
    
    # Distribute across columns
    cols = [col1, col2, col3]

    count = 0
    for cat in categories:
        # Case insensitive match attempt if exact not found
        if cat not in available_cats:
             # Try to find a match
             match = next((c for c in available_cats if c.upper() == cat), None)
             if match:
                 cat = match
        
        if cat in available_cats:
            current_col = cols[count % 3]
            idx = create_selectbox(current_col, cat)
            if idx is not None:
                selected_indices.append(idx)
            count += 1
        else:
            # Maybe dynamic categories?
            pass

    # Run Simulation
    st.header("2. Run Simulation")

    if st.button("🚀 Run Risk Analysis", type="primary"):
        with st.spinner("Simulating 1,000+ futures..."):
            results = logic.run_simulation(selected_indices, n_iterations, components_df, events_df)
            
        final_costs = results['final_costs']
        base_price = results['base_price']
        
        # Safe survival calculation
        if len(final_costs) > 0:
            survival_count = np.sum(final_costs <= budget_limit)
            survival_rate = (survival_count / n_iterations) * 100
            
            # Metrics
            st.subheader("Simulation Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Base Price", f"${base_price:,.2f}")
            m2.metric("Median Projected Cost", f"${results['median_cost']:,.2f}", delta=f"{results['median_cost']-base_price:,.2f} Risk Premium", delta_color="inverse")
            m3.metric("Worst Case Scenario", f"${results['worst_case']:,.2f}")
            m4.metric("Budget Survival Rate", f"{survival_rate:.1f}%", delta=f"{survival_rate-100 if survival_rate < 100 else 0:.1f}% Risk", delta_color="normal")
            
            # Visualization
            st.subheader("Cost Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Histogram
            n, bins, patches = ax.hist(final_costs, bins=50, color='skyblue', alpha=0.7, edgecolor='black', label='Simulated Costs')
            
            # Lines for Budget and Base
            ax.axvline(base_price, color='green', linestyle='dashed', linewidth=2, label=f'Base Price (${base_price})')
            ax.axvline(budget_limit, color='red', linestyle='dashed', linewidth=2, label=f'Budget Limit (${budget_limit})')
            
            # Highlight failure zone
            if np.max(final_costs) > budget_limit:
                ax.axvspan(budget_limit, np.max(final_costs), color='red', alpha=0.1, label='Over Budget Zone')

            ax.set_xlabel('Total Cost ($)')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)
            
            # Explanation
            st.info(f"""
            **Analysis**: In {n_iterations} simulated futures, your build stayed under the ${budget_limit} budget **{survival_rate:.1f}%** of the time.
            The "Risk Premium" indicates the median extra cost you might expect to pay due to market volatility.
            """)
        else:
            st.warning("Simulation returned no results. Please check your Component selection.")
