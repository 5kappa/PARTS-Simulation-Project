import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logic
import sys
from streamlit.web import cli as stcli

# Page Config (Must be the first Streamlit command)
st.set_page_config(page_title="P.A.R.T.S.", layout="wide")

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

st.title("🖥️ Welcome to P.A.R.T.S: P.C. Assembly & Rig Tracking Simulator!")
st.markdown("""
This tool uses **Monte Carlo Simulation** to test if your PC build budget can survive future market volatility.
It simulates **1,000+ market scenarios** (AI shortages, crypto booms, supply gluts) to estimate probabilistic costs.
""")

# Sidebar
st.sidebar.header("Simulation Settings")
budget_limit = st.sidebar.number_input("Max Budget ($)", min_value=500, max_value=10000, value=2000, step=100)
n_iterations = st.sidebar.slider("Simulation Iterations", min_value=100, max_value=10000, value=2000, step=100)
time_horizon = st.sidebar.slider("Time Horizon (Months)", min_value=1, max_value=24, value=12, help="Forecasting period. Longer time = Higher probability of events occurring.")

st.sidebar.markdown("---")
if time_horizon != 12:
    st.sidebar.info(f"**Scaling**: Event probabilities are adjusted for a **{time_horizon}-month** window.")
else:
    st.sidebar.info("**Time Horizon**: Models market volatility for the **next 12 months** (Annualized Baseline).")

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Manage Builds")

# Load saved rigs
saved_rigs = logic.load_builds()

# Save Logic
save_name = st.sidebar.text_input("Name New Rig", placeholder="e.g. Dream Build 2025")
if st.sidebar.button("Save Current Selection"):
    if 'current_indices' in st.session_state and st.session_state.current_indices:
        if save_name:
            if logic.save_build(save_name, st.session_state.current_indices):
                st.sidebar.success(f"Saved '{save_name}'!")
                # Rerun to update the dropdown immediately
                st.rerun()
            else:
                st.sidebar.error("Save failed.")
        else:
            st.sidebar.warning("Please enter a name.")
    else:
        st.sidebar.warning("No components selected yet.")

# Load/Delete Logic
if saved_rigs:
    load_name = st.sidebar.selectbox(
        "Load/Delete Saved Rig", 
        options=list(saved_rigs.keys()), 
        index=None, 
        placeholder="Select a rig..."
    )
    
    if load_name:
        ld_col1, ld_col2 = st.sidebar.columns(2)
        
        if ld_col1.button("📂 Load", type="primary", use_container_width=True):
            loaded = saved_rigs[load_name]
            # Update session state for specific widgets to force them to change
            for idx in loaded:
                if idx in components_df.index:
                    cat = components_df.loc[idx, 'Category']
                    st.session_state[f"select_{cat}"] = idx
            
            st.session_state.loaded_indices = loaded # Keep track
            st.rerun()
            
        if ld_col2.button("🗑️ Delete", type="secondary", use_container_width=True):
            if logic.delete_build(load_name):
                st.sidebar.success(f"Deleted '{load_name}'")
                st.rerun()
            else:
                st.sidebar.error("Delete failed")

# Comparison Toggle
st.sidebar.markdown("---")
comparison_mode = st.sidebar.checkbox("🔀 Comparison Mode", value=False)

if not events_df.empty:
    st.sidebar.header("Market Events Preview")
    st.sidebar.dataframe(events_df[['EventName', 'Probability', 'Multiplier']], hide_index=True)
else:
    st.error("Events data could not be loaded!")

# ==========================================
# COMPARISON MODE UI
# ==========================================
if comparison_mode:
    st.header("🔀 Build Comparison Mode")
    
    if not saved_rigs:
        st.warning("You need to save at least one rig before you can compare!")
    else:
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.subheader("Build A")
            rig_a_name = st.selectbox("Select Rig A", options=["Current Selection"] + list(saved_rigs.keys()))
            
            if rig_a_name == "Current Selection":
                if 'current_indices' in st.session_state:
                    indices_a = st.session_state.current_indices
                else:
                    indices_a = []
            else:
                indices_a = saved_rigs[rig_a_name]
            
            # mini display
            if indices_a:
                cost_a = components_df.loc[indices_a, 'BasePrice'].sum()
                st.metric("Base Price A", f"${cost_a:,.2f}")
            
        with comp_col2:
            st.subheader("Build B")
            # Default to second rig if available
            default_b_idx = 1 if len(saved_rigs) > 1 else 0
            rig_b_name = st.selectbox("Select Rig B", options=list(saved_rigs.keys()), index=default_b_idx)
            indices_b = saved_rigs[rig_b_name]
            
            if indices_b:
                cost_b = components_df.loc[indices_b, 'BasePrice'].sum()
                st.metric("Base Price B", f"${cost_b:,.2f}")
        
        st.markdown("---")
        
        if st.button("🚀 Compare Builds", type="primary"):
            if not indices_a or not indices_b:
                st.error("Please ensure both builds have components selected.")
            else:
                with st.spinner("Simulating both futures..."):
                    res_a = logic.run_simulation(indices_a, n_iterations, components_df, events_df, time_horizon)
                    res_b = logic.run_simulation(indices_b, n_iterations, components_df, events_df, time_horizon)
                
                # METRICS COMPARISON
                st.subheader("Risk Metrics Comparison")
                
                # Calculate metrics
                surv_a = (np.sum(res_a['final_costs'] <= budget_limit) / n_iterations) * 100
                surv_b = (np.sum(res_b['final_costs'] <= budget_limit) / n_iterations) * 100
                
                comp_data = {
                    "Metric": ["Median Component Cost", "95% VaR (Worst Case)", "Budget Survival Rate"],
                    f"{rig_a_name}": [
                        f"${res_a['median_cost']:,.2f}", 
                        f"${res_a['var_95']:,.2f}", 
                        f"{surv_a:.1f}%"
                    ],
                    f"{rig_b_name}": [
                        f"${res_b['median_cost']:,.2f}", 
                        f"${res_b['var_95']:,.2f}", 
                        f"{surv_b:.1f}%"
                    ],
                    "Difference": [
                        f"${res_b['median_cost'] - res_a['median_cost']:,.2f}",
                        f"${res_b['var_95'] - res_a['var_95']:,.2f}",
                        f"{surv_b - surv_a:.1f}%"
                    ]
                }
                
                comp_df = pd.DataFrame(comp_data)
                
                # Apply styling for clearer reading
                styled_df = comp_df.style.set_properties(**{
                    'font-size': '18px',
                    'text-align': 'center'
                }).set_table_styles([
                    {'selector': 'th', 'props': [('font-size', '18px'), ('text-align', 'center')]}
                ])
                
                st.dataframe(styled_df, hide_index=True, use_container_width=True)
                
                # OVERLAPPING HISTOGRAM
                st.subheader("Probability Distribution Comparison")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                ax.hist(res_a['final_costs'], bins=50, alpha=0.5, label=f'Rig A: {rig_a_name}', color='blue')
                ax.hist(res_b['final_costs'], bins=50, alpha=0.5, label=f'Rig B: {rig_b_name}', color='orange')
                
                ax.axvline(budget_limit, color='red', linestyle='--', linewidth=2, label='Budget Limit')
                
                ax.set_xlabel('Total Cost ($)')
                ax.set_ylabel('Frequency')
                ax.legend()
                st.pyplot(fig)


# ==========================================
# STANDARD MODE UI
# ==========================================
else:
    # Main UI - Component Selection
    st.header("1. Select Your Components")

    if components_df.empty:
        st.error("Components data could not be loaded! Please check components.csv.")
        selected_indices = []
    else:
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

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
            
            # We use a specific key for state management
            key_name = f"select_{category_name}"
            
            # Ensure default (0) if not set in state
            # But if we just loaded, state is already set by the Load Button logic above.
            # Streamlit handles 'index' vs 'session_state' priority: state wins if key exists.
            
            selected_idx = column.selectbox(
                f"Select {category_name}", 
                options=options, 
                format_func=lambda x: labels[x],
                index=0, 
                key=key_name
            )
            return selected_idx

        selected_indices = []
        # Standard Categories to look for (Standardize headers)
        # We use the categories present in the CSV
        available_cats = components_df['Category'].unique()
        
        count = 0
        for cat in available_cats:
            current_col = cols[count % 3]
            idx = create_selectbox(current_col, cat)
            if idx is not None:
                 selected_indices.append(idx)
            count += 1

        # Store current selection in session state for Saving
        st.session_state.current_indices = selected_indices

        # Display Current Total Price
        if selected_indices:
            current_total = components_df.loc[selected_indices, 'BasePrice'].sum()
            st.subheader(f"💰 Current Build Total: :green[${current_total:,.2f}]")
        
        # Run Simulation
        st.header("2. Run Simulation")

        if st.button("🚀 Run Risk Analysis", type="primary"):
            with st.spinner(f"Simulating 1,000+ futures over {time_horizon} months..."):
                results = logic.run_simulation(selected_indices, n_iterations, components_df, events_df, time_horizon)
                
            final_costs = results['final_costs']
            base_price = results['base_price']
            
            # Safe survival calculation
            if len(final_costs) > 0:
                survival_count = np.sum(final_costs <= budget_limit)
                survival_rate = (survival_count / n_iterations) * 100
                
                # Metrics
                st.subheader("Simulation Results")
                metric_cols = st.columns(4)
                
                # 1. Base Price
                metric_cols[0].metric("Base Price", f"${base_price:,.2f}")
                
                # 2. Median Projected Cost (Delta: Inflation)
                inflation = results['median_cost'] - base_price
                
                # Format delta string and color
                if inflation > 0:
                    infl_str = f"+${inflation:,.2f} Inflation"
                    infl_color = "inverse"  # Red (Cost up)
                elif inflation < 0:
                    infl_str = f"-${abs(inflation):,.2f} Inflation"
                    infl_color = "inverse"  # Green (Cost down - Inverse of standard growth)
                else:
                    infl_str = "$0.00 Inflation"
                    infl_color = "off"      # Neutral
                
                metric_cols[1].metric(
                    "Median Projected Cost", 
                    f"${results['median_cost']:,.2f}", 
                    delta=infl_str,
                    delta_color=infl_color,
                    help="The most likely cost of your build. Delta shows the expected inflation over Base Price."
                )
                
                # 3. Worst Case Scenario (Delta: Risk Premium)
                # Risk Premium = 90th percentile - Median
                risk_premium = results['cost_90th'] - results['median_cost']
                metric_cols[2].metric(
                    "Worst Case Scenario", 
                    f"${results['worst_case']:,.2f}", 
                    delta=f"+${risk_premium:,.2f} Risk Premium",
                    delta_color="inverse",
                    help="The absolute maximum cost simulation found. Delta shows the 'Risk Premium' (90th Percentile - Median) you should buffer."
                )
                
                # 4. Budget Survival
                metric_cols[3].metric(
                    "Budget Survival Rate", 
                    f"{survival_rate:.1f}%", 
                    delta=f"{survival_rate-100 if survival_rate < 100 else 0:.1f}% Risk", 
                    delta_color="normal"
                )
                
                # Create Tabs
                tab1, tab2 = st.tabs(["📊 Probability Distribution", "🌪️ Sensitivity Analysis"])
                
                with tab1:
                    st.subheader("Cost Probability Distribution")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Histogram
                    n, bins, patches = ax.hist(final_costs, bins=50, color='skyblue', alpha=0.7, edgecolor='black', label='Simulated Costs')
                    
                    # Lines for Budget, Base, and 90th
                    ax.axvline(base_price, color='green', linestyle='dashed', linewidth=2, label=f'Base Price (${base_price})')
                    ax.axvline(results['cost_90th'], color='orange', linestyle='--', linewidth=2, label=f'90th Pct (${results["cost_90th"]:,.0f})')
                    ax.axvline(results['var_95'], color='red', linestyle='-.', linewidth=2, label=f'VaR 95% (${results["var_95"]:,.0f})')
                    ax.axvline(budget_limit, color='black', linestyle='dotted', linewidth=2, label=f'Budget (${budget_limit})')
                    
                    # Highlight failure zone
                    if np.max(final_costs) > budget_limit:
                        ax.axvspan(budget_limit, np.max(final_costs), color='red', alpha=0.1, label='Over Budget')

                    ax.set_xlabel('Total Cost ($)')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    st.pyplot(fig)
                    
                    st.info(f"""
                    **Analysis**: 
                    *   **Projected Inflation**: The median outcome is **${inflation:,.2f}** higher than base price.
                    *   **Risk Premium**: To be 90% safe, you need a buffer of **${risk_premium:,.2f}** on top of the median.
                    *   **Value at Risk (95%)**: We are **95% confident** your cost will not exceed **${results['var_95']:,.2f}**.
                    """)
                    
                with tab2:
                    st.subheader("Top Risk Drivers (Sensitivity Analysis)")
                    st.markdown("This chart shows which events define your financial risk. **Red bars** mean the event increases cost, **Green bars** mean it saves money.")
                    
                    # Calculate
                    sensitivity_df = logic.calculate_sensitivity(results, events_df)
                    
                    if not sensitivity_df.empty:
                        # Take top 10
                        top_risks = sensitivity_df.head(10).iloc[::-1] # Reverse for horizontal bar chart
                        
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        
                        # Color logic: Red for positive impact (bad for budget), Green for negative (good)
                        colors = ['#ff4b4b' if x > 0 else '#21c354' for x in top_risks['Impact ($)']]
                        
                        bars = ax2.barh(top_risks['Event'], top_risks['Impact ($)'], color=colors)
                        
                        ax2.set_xlabel('Average Impact on Total Bundle Cost ($)')
                        ax2.set_title('Event Impact (Tornado Chart)')
                        ax2.grid(axis='x', linestyle='--', alpha=0.7)
                        
                        # Add Labels
                        ax2.bar_label(bars, fmt='$%.0f', padding=3)
                        
                        st.pyplot(fig2)
                    else:
                        st.warning("Not enough variance to calculate sensitivity.")
            else:
                st.warning("Simulation returned no results. Please check your Component selection.")
