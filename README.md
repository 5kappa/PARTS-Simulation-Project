# P.A.R.T.S.: P.C. Assembly and Rig Tracking Simulator

A web-based Monte Carlo simulation tool that tests the resilience of a PC build budget against hyper-volatile market conditions. It specifically models high-precision market shocks (e.g., "AI Arms Race" targeting DDR5 RAM) to estimate the probability of your build exceeding its budget.

## Project Goal
To demonstrate **Stochastic Modeling** and **Sensitivity Analysis** by simulating 1,000+ future market scenarios and their impact on specific PC components.

## Key Features
- **Monte Carlo Risk Engine**: Instantly simulates 1,000s of market futures to estimate specific financial risks (VaR, Median Cost).
- **Comparison Mode**: Run side-by-side simulations of two different builds (e.g., "Budget vs. Dream Rig") to compare their survival rates and cost distributions.
- **Rig Management**: Save, Load, and Delete your build configurations locally.
- **Dynamic Time Horizon**: Scale risk probability based on your purchasing timeframe (1-24 months).
- **Interactive Visualizations**:
    - **Probability Histograms**: See the exact likelihood of price outcomes.
    - **Tornado Charts (Sensitivity Analysis)**: Identify which specific events (e.g., "Taiwan Strait Tension") are driving your risk.
- **High-Precision Targeting**: Market events target specific ID tags (e.g., "DDR5 Shortage" only affects components with volatility score `0.95`).

## Setup and Installation for Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/5kappa/PARTS-Simulation-Project.git
   cd PARTS-Simulation-Project
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv .venv
   
   # Activate on Windows:
   .venv\Scripts\activate
   
   # Activate on Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

You can run the application directly from your terminal:

```bash
streamlit run app.py
```

Alternatively, if you are in the virtual environment, you can simply run:
```bash
python app.py
```

## Data Structure
- `data/components.csv`: List of PC parts with Base Prices and Volatility Scores.
- `data/events.csv`: List of market events with Probabilities, Multipliers, and Targeting rules.
