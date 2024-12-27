import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter  # Imported for percentage formatting

# Set page configuration
st.set_page_config(page_title="Poker Tournament Monte Carlo Simulator", layout="wide")

# Title and Description
st.title("Poker Tournament Monte Carlo Simulator")
st.markdown("""
Welcome to the **Poker Tournament Monte Carlo Simulator**! This application allows you to simulate Spin and Go poker tournaments to analyze potential outcomes based on a player's Chip EV (expected chips won per game). Adjust the simulation settings below to explore different scenarios and understand your potential returns and risks.
""")  # Added Description under the main title

# Tournament Details
st.header("Tournament Details")

# Define initial chips and target chips (unchanged)
initial_chips = 500
target_chips = 1500

# Define prize distributions for all stakes (unchanged)
prize_distributions = {
    "$1 Stake": [
        {"Prize": "Prize 1", "Probability": 0.4772497, "Prize Pool": 2, "P1": 2, "P2": 0, "P3": 0},
        {"Prize": "Prize 2", "Probability": 0.3968502, "Prize Pool": 3, "P1": 3, "P2": 0, "P3": 0},
        {"Prize": "Prize 3", "Probability": 0.09, "Prize Pool": 4, "P1": 4, "P2": 0, "P3": 0},
        {"Prize": "Prize 4", "Probability": 0.025, "Prize Pool": 5, "P1": 5, "P2": 0, "P3": 0},
        {"Prize": "Prize 5", "Probability": 0.01, "Prize Pool": 10, "P1": 8, "P2": 2, "P3": 0},
        {"Prize": "Prize 6", "Probability": 0.00075, "Prize Pool": 25, "P1": 20, "P2": 3, "P3": 2},
        {"Prize": "Prize 7", "Probability": 0.0001, "Prize Pool": 50, "P1": 40, "P2": 6, "P3": 4},
        {"Prize": "Prize 8", "Probability": 0.00005, "Prize Pool": 100, "P1": 80, "P2": 12, "P3": 8},
        {"Prize": "Prize 9", "Probability": 0.0000001, "Prize Pool": 12000, "P1": 10000, "P2": 1200, "P3": 800},
    ],
    "$2 Stake": [
        {"Prize": "Prize 1", "Probability": 0.4772497, "Prize Pool": 4, "P1": 4, "P2": 0, "P3": 0},
        {"Prize": "Prize 2", "Probability": 0.3968502, "Prize Pool": 6, "P1": 6, "P2": 0, "P3": 0},
        {"Prize": "Prize 3", "Probability": 0.09, "Prize Pool": 8, "P1": 8, "P2": 0, "P3": 0},
        {"Prize": "Prize 4", "Probability": 0.025, "Prize Pool": 10, "P1": 10, "P2": 0, "P3": 0},
        {"Prize": "Prize 5", "Probability": 0.01, "Prize Pool": 20, "P1": 16, "P2": 4, "P3": 0},
        {"Prize": "Prize 6", "Probability": 0.00075, "Prize Pool": 50, "P1": 40, "P2": 6, "P3": 4},
        {"Prize": "Prize 7", "Probability": 0.0001, "Prize Pool": 100, "P1": 80, "P2": 12, "P3": 8},
        {"Prize": "Prize 8", "Probability": 0.00005, "Prize Pool": 200, "P1": 160, "P2": 24, "P3": 16},
        {"Prize": "Prize 9", "Probability": 0.0000001, "Prize Pool": 24000, "P1": 20000, "P2": 2400, "P3": 1600},
    ],
    "$5 Stake": [
        {"Prize": "Prize 1", "Probability": 0.5000497, "Prize Pool": 10, "P1": 10, "P2": 0, "P3": 0},
        {"Prize": "Prize 2", "Probability": 0.3740502, "Prize Pool": 15, "P1": 15, "P2": 0, "P3": 0},
        {"Prize": "Prize 3", "Probability": 0.09, "Prize Pool": 20, "P1": 20, "P2": 0, "P3": 0},
        {"Prize": "Prize 4", "Probability": 0.025, "Prize Pool": 25, "P1": 25, "P2": 0, "P3": 0},
        {"Prize": "Prize 5", "Probability": 0.01, "Prize Pool": 50, "P1": 40, "P2": 10, "P3": 0},
        {"Prize": "Prize 6", "Probability": 0.00075, "Prize Pool": 125, "P1": 100, "P2": 15, "P3": 10},
        {"Prize": "Prize 7", "Probability": 0.0001, "Prize Pool": 250, "P1": 200, "P2": 30, "P3": 20},
        {"Prize": "Prize 8", "Probability": 0.00005, "Prize Pool": 500, "P1": 400, "P2": 60, "P3": 40},
        {"Prize": "Prize 9", "Probability": 0.0000001, "Prize Pool": 1200000, "P1": 1000000, "P2": 120000, "P3": 80000},
    ],
    "$10 Stake": [
        {"Prize": "Prize 1", "Probability": 0.4472497, "Prize Pool": 20, "P1": 20, "P2": 0, "P3": 0},
        {"Prize": "Prize 2", "Probability": 0.4268502, "Prize Pool": 30, "P1": 30, "P2": 0, "P3": 0},
        {"Prize": "Prize 3", "Probability": 0.09, "Prize Pool": 40, "P1": 40, "P2": 0, "P3": 0},
        {"Prize": "Prize 4", "Probability": 0.025, "Prize Pool": 50, "P1": 50, "P2": 0, "P3": 0},
        {"Prize": "Prize 5", "Probability": 0.01, "Prize Pool": 100, "P1": 80, "P2": 20, "P3": 0},
        {"Prize": "Prize 6", "Probability": 0.00075, "Prize Pool": 250, "P1": 200, "P2": 30, "P3": 20},
        {"Prize": "Prize 7", "Probability": 0.0001, "Prize Pool": 500, "P1": 400, "P2": 60, "P3": 40},
        {"Prize": "Prize 8", "Probability": 0.00005, "Prize Pool": 1000, "P1": 800, "P2": 120, "P3": 80},
        {"Prize": "Prize 9", "Probability": 0.0000001, "Prize Pool": 120000, "P1": 100000, "P2": 12000, "P3": 8000},
    ],
    "$25 Stake": [
        {"Prize": "Prize 1", "Probability": 0.4172497, "Prize Pool": 50, "P1": 50, "P2": 0, "P3": 0},
        {"Prize": "Prize 2", "Probability": 0.4568502, "Prize Pool": 75, "P1": 75, "P2": 0, "P3": 0},
        {"Prize": "Prize 3", "Probability": 0.09, "Prize Pool": 100, "P1": 100, "P2": 0, "P3": 0},
        {"Prize": "Prize 4", "Probability": 0.025, "Prize Pool": 125, "P1": 125, "P2": 0, "P3": 0},
        {"Prize": "Prize 5", "Probability": 0.01, "Prize Pool": 250, "P1": 200, "P2": 50, "P3": 0},
        {"Prize": "Prize 6", "Probability": 0.00075, "Prize Pool": 625, "P1": 500, "P2": 75, "P3": 50},
        {"Prize": "Prize 7", "Probability": 0.0001, "Prize Pool": 1250, "P1": 1000, "P2": 150, "P3": 100},
        {"Prize": "Prize 8", "Probability": 0.00005, "Prize Pool": 2500, "P1": 2000, "P2": 300, "P3": 200},
        {"Prize": "Prize 9", "Probability": 0.0000001, "Prize Pool": 300000, "P1": 250000, "P2": 30000, "P3": 20000},
    ],
    "$50 Stake": [
        {"Prize": "Prize 1", "Probability": 0.4172497, "Prize Pool": 100, "P1": 100, "P2": 0, "P3": 0},
        {"Prize": "Prize 2", "Probability": 0.4568502, "Prize Pool": 150, "P1": 150, "P2": 0, "P3": 0},
        {"Prize": "Prize 3", "Probability": 0.09, "Prize Pool": 200, "P1": 200, "P2": 0, "P3": 0},
        {"Prize": "Prize 4", "Probability": 0.025, "Prize Pool": 250, "P1": 250, "P2": 0, "P3": 0},
        {"Prize": "Prize 5", "Probability": 0.01, "Prize Pool": 500, "P1": 400, "P2": 100, "P3": 0},
        {"Prize": "Prize 6", "Probability": 0.00075, "Prize Pool": 1250, "P1": 1000, "P2": 150, "P3": 100},
        {"Prize": "Prize 7", "Probability": 0.0001, "Prize Pool": 2500, "P1": 2000, "P2": 300, "P3": 200},
        {"Prize": "Prize 8", "Probability": 0.00005, "Prize Pool": 5000, "P1": 4000, "P2": 600, "P3": 400},
        {"Prize": "Prize 9", "Probability": 0.0000001, "Prize Pool": 600000, "P1": 500000, "P2": 60000, "P3": 40000},
    ],
    "$100 Stake": [
        {"Prize": "Prize 1", "Probability": 0.3872497, "Prize Pool": 200, "P1": 200, "P2": 0, "P3": 0},
        {"Prize": "Prize 2", "Probability": 0.4868502, "Prize Pool": 300, "P1": 300, "P2": 0, "P3": 0},
        {"Prize": "Prize 3", "Probability": 0.09, "Prize Pool": 400, "P1": 400, "P2": 0, "P3": 0},
        {"Prize": "Prize 4", "Probability": 0.025, "Prize Pool": 500, "P1": 500, "P2": 0, "P3": 0},
        {"Prize": "Prize 5", "Probability": 0.01, "Prize Pool": 1000, "P1": 800, "P2": 200, "P3": 0},
        {"Prize": "Prize 6", "Probability": 0.00075, "Prize Pool": 2500, "P1": 2000, "P2": 300, "P3": 200},
        {"Prize": "Prize 7", "Probability": 0.0001, "Prize Pool": 5000, "P1": 4000, "P2": 600, "P3": 400},
        {"Prize": "Prize 8", "Probability": 0.00005, "Prize Pool": 10000, "P1": 8000, "P2": 1200, "P3": 800},
        {"Prize": "Prize 9", "Probability": 0.0000001, "Prize Pool": 1200000, "P1": 1000000, "P2": 120000, "P3": 80000},
    ],
    "$250 Stake": [
        {"Prize": "Prize 1", "Probability": 0.3865297, "Prize Pool": 500, "P1": 500, "P2": 0, "P3": 0},
        {"Prize": "Prize 2", "Probability": 0.4875702, "Prize Pool": 750, "P1": 750, "P2": 0, "P3": 0},
        {"Prize": "Prize 3", "Probability": 0.09, "Prize Pool": 1000, "P1": 1000, "P2": 0, "P3": 0},
        {"Prize": "Prize 4", "Probability": 0.025, "Prize Pool": 1250, "P1": 1250, "P2": 0, "P3": 0},
        {"Prize": "Prize 5", "Probability": 0.01, "Prize Pool": 2500, "P1": 2000, "P2": 500, "P3": 0},
        {"Prize": "Prize 6", "Probability": 0.00075, "Prize Pool": 6250, "P1": 5000, "P2": 750, "P3": 500},
        {"Prize": "Prize 7", "Probability": 0.0001, "Prize Pool": 12500, "P1": 10000, "P2": 1500, "P3": 1000},
        {"Prize": "Prize 8", "Probability": 0.00005, "Prize Pool": 25000, "P1": 20000, "P2": 3000, "P3": 2000},
        {"Prize": "Prize 9", "Probability": 0.0000001, "Prize Pool": 1200000, "P1": 1000000, "P2": 120000, "P3": 80000},
    ],
    "$500 Stake": [
        {"Prize": "Prize 1", "Probability": 0.3862897, "Prize Pool": 1000, "P1": 1000, "P2": 0, "P3": 0},
        {"Prize": "Prize 2", "Probability": 0.4878102, "Prize Pool": 1500, "P1": 1500, "P2": 0, "P3": 0},
        {"Prize": "Prize 3", "Probability": 0.09, "Prize Pool": 2000, "P1": 2000, "P2": 0, "P3": 0},
        {"Prize": "Prize 4", "Probability": 0.025, "Prize Pool": 2500, "P1": 2500, "P2": 0, "P3": 0},
        {"Prize": "Prize 5", "Probability": 0.01, "Prize Pool": 5000, "P1": 4000, "P2": 1000, "P3": 0},
        {"Prize": "Prize 6", "Probability": 0.00075, "Prize Pool": 12500, "P1": 10000, "P2": 1500, "P3": 1000},
        {"Prize": "Prize 7", "Probability": 0.0001, "Prize Pool": 25000, "P1": 20000, "P2": 3000, "P3": 2000},
        {"Prize": "Prize 8", "Probability": 0.00005, "Prize Pool": 50000, "P1": 40000, "P2": 6000, "P3": 4000},
        {"Prize": "Prize 9", "Probability": 0.0000001, "Prize Pool": 1200000, "P1": 1000000, "P2": 120000, "P3": 80000},
    ],
    "$1000 Stake": [
        {"Prize": "Prize 1", "Probability": 0.3861697, "Prize Pool": 2000, "P1": 2000, "P2": 0, "P3": 0},
        {"Prize": "Prize 2", "Probability": 0.4879302, "Prize Pool": 3000, "P1": 3000, "P2": 0, "P3": 0},
        {"Prize": "Prize 3", "Probability": 0.09, "Prize Pool": 4000, "P1": 4000, "P2": 0, "P3": 0},
        {"Prize": "Prize 4", "Probability": 0.025, "Prize Pool": 5000, "P1": 5000, "P2": 0, "P3": 0},
        {"Prize": "Prize 5", "Probability": 0.01, "Prize Pool": 10000, "P1": 8000, "P2": 2000, "P3": 0},
        {"Prize": "Prize 6", "Probability": 0.00075, "Prize Pool": 25000, "P1": 20000, "P2": 3000, "P3": 2000},
        {"Prize": "Prize 7", "Probability": 0.0001, "Prize Pool": 50000, "P1": 40000, "P2": 6000, "P3": 4000},
        {"Prize": "Prize 8", "Probability": 0.00005, "Prize Pool": 100000, "P1": 80000, "P2": 12000, "P3": 8000},
        {"Prize": "Prize 9", "Probability": 0.0000001, "Prize Pool": 1200000, "P1": 1000000, "P2": 120000, "P3": 80000},
    ]
}

# User selects the stake
st.subheader("Select Stake")
stake_options = ["$1 Stake", "$2 Stake", "$5 Stake", "$10 Stake", "$25 Stake","$50 Stake","$100 Stake","$250 Stake","$500 Stake","$1000 Stake"]
stake_default = "$10 Stake"
stake = st.selectbox("Choose your stake:", options=stake_options, index=stake_options.index(stake_default))

# Set buy_in and select the corresponding prize distribution
if stake == "$1 Stake":
    buy_in = 1
    selected_prizes = prize_distributions["$1 Stake"]
elif stake == "$2 Stake":
    buy_in = 2
    selected_prizes = prize_distributions["$2 Stake"]
elif stake == "$5 Stake":
    buy_in = 5
    selected_prizes = prize_distributions["$5 Stake"]
elif stake == "$10 Stake":
    buy_in = 10
    selected_prizes = prize_distributions["$10 Stake"]
elif stake == "$25 Stake":
    buy_in = 25
    selected_prizes = prize_distributions["$25 Stake"]
elif stake == "$50 Stake":
    buy_in = 50
    selected_prizes = prize_distributions["$50 Stake"]
elif stake == "$100 Stake":
    buy_in = 100
    selected_prizes = prize_distributions["$100 Stake"]
elif stake == "$250 Stake":
    buy_in = 250
    selected_prizes = prize_distributions["$250 Stake"] 
elif stake == "$500 Stake":
    buy_in = 500
    selected_prizes = prize_distributions["$500 Stake"]
elif stake == "$1000 Stake":
    buy_in = 1000
    selected_prizes = prize_distributions["$1000 Stake"]           

# Convert selected prizes to DataFrame
prize_df = pd.DataFrame(selected_prizes)
prize_df = prize_df[['Prize', 'Probability', 'Prize Pool', 'P1', 'P2', 'P3']]

# Display Buy-In Details
st.subheader("Buy-In Details")
st.write(f"**Stake Selected:** {stake}")
st.write(f"**Buy-In per Player:** ${buy_in}")
st.write(f"Each player starts with **{initial_chips}** chips.")
st.write(f"The player who collects all **{target_chips}** chips finishes first.")

# Display Prize Distribution
st.subheader("Prize Distribution")
st.table(prize_df.style.format({
    "Probability": "{:.7f}",
    "Prize Pool": "${}",
    "P1": "${}",
    "P2": "${}",
    "P3": "${}"
}))

# User Inputs for Simulation
st.header("Simulation Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    expected_chips = st.number_input(
        "Expected Chips per Tournament",
        min_value=-500.0,
        max_value=1000.0,
        value=61.0,  # Updated default
        step=1.0,
        help="Average number of chips you expect to win per tournament."
    )
with col2:
    num_tournaments = st.number_input(
        "Number of Tournaments",
        min_value=1,
        max_value=100000,
        value=5000,  # Updated default
        step=5000,
        help="Total number of tournaments to simulate."
    )
with col3:
    num_simulations = st.number_input(
        "Number of Simulations",
        min_value=1,
        max_value=10000,
        value=100,  # Updated default
        step=100,
        help="Number of Monte Carlo simulations to run."
    )

# Validate expected_chips
prob_first = (expected_chips + initial_chips) / target_chips
if prob_first > 1:
    st.warning("The probability of finishing first exceeds 1. Please adjust your expected chips per tournament.")
    st.stop()
prob_first = np.clip(prob_first, 0, 1)  # Ensure probability is between 0 and 1

prob_not_first = 1 - prob_first
prob_second = prob_not_first * 0.5
prob_third = prob_not_first * 0.5

# Prepare Prize Pools and Payouts
prize_probs = prize_df['Probability'].values
cumulative_probs = np.cumsum(prize_probs)

# Ensure the last cumulative probability is exactly 1 to avoid indexing issues
cumulative_probs[-1] = 1.0

# Extract Payouts as separate NumPy arrays for vectorized indexing
P1 = prize_df['P1'].values  # Shape: (num_prizes,)
P2 = prize_df['P2'].values
P3 = prize_df['P3'].values

# Simulation
st.header("Monte Carlo Simulation Results")

if st.button("Run Simulation"):
    with st.spinner("Running simulations..."):
        # Initialize arrays to store results
        cumulative_profits = np.zeros((num_simulations, num_tournaments))
        games_finished_first = np.zeros(num_simulations)
        max_drawdowns = np.zeros(num_simulations)
        max_drawdowns_buyins = np.zeros(num_simulations)
        
        rng = np.random.default_rng()
        
        # Pre-generate random numbers for finish positions
        finish_randoms = rng.random((num_simulations, num_tournaments))
        finish_positions = np.where(
            finish_randoms < prob_first, 1,
            np.where(finish_randoms < prob_first + prob_second, 2, 3)
        )
        
        # Pre-generate random numbers for prize selection
        prize_randoms = rng.random((num_simulations, num_tournaments))
        # Find prize indices based on cumulative probabilities
        prize_indices = np.searchsorted(cumulative_probs, prize_randoms)
        # Handle edge cases where random number is exactly 1.0
        prize_indices = np.clip(prize_indices, 0, len(prize_probs)-1)
        
        # Vectorized prize payout selection
        selected_P1 = P1[prize_indices]  # Shape: (num_simulations, num_tournaments)
        selected_P2 = P2[prize_indices]
        selected_P3 = P3[prize_indices]
        
        # Stack payouts for easy indexing
        # Shape: (num_simulations, num_tournaments, 3)
        selected_prizes = np.stack([selected_P1, selected_P2, selected_P3], axis=2)
        
        # Assign profits based on finish positions
        # Create a mask for each finishing position
        mask_first = finish_positions == 1  # Shape: (num_simulations, num_tournaments)
        mask_second = finish_positions == 2
        mask_third = finish_positions == 3
        
        # Initialize user profits with -buy_in (since each tournament requires a buy-in)
        user_profits = np.full((num_simulations, num_tournaments), -buy_in, dtype=np.float64)
        
        # Add prize money based on finishing position
        user_profits += selected_prizes[:, :, 0] * mask_first
        user_profits += selected_prizes[:, :, 1] * mask_second
        user_profits += selected_prizes[:, :, 2] * mask_third
        
        # **New Addition: Identify Zero Payout Tournaments**
        zero_payout = np.isclose(user_profits + buy_in, 0)  # True where payout was $0
        zero_payout_counts = zero_payout.sum(axis=1)  # Number of zero payouts per simulation
        zero_payout_percentage = (zero_payout_counts / num_tournaments) * 100  # Percentage per simulation
        
        # Calculate cumulative profits
        cumulative_profits = np.cumsum(user_profits, axis=1)
        
        # Calculate max drawdown for each simulation
        # Max drawdown is the maximum peak-to-trough decline
        running_max = np.maximum.accumulate(cumulative_profits, axis=1)
        drawdowns = running_max - cumulative_profits
        max_drawdowns = np.max(drawdowns, axis=1)
        max_drawdowns_buyins = max_drawdowns / buy_in
        
        # Count games finished first
        games_finished_first = np.sum(mask_first, axis=1)
        
        # **New Addition: Calculate Percentage of Games Finished 1st**
        games_finished_first_percentage = (games_finished_first / num_tournaments) * 100
        avg_games_first_percentage = np.mean(games_finished_first_percentage)
        max_games_first_percentage = np.max(games_finished_first_percentage)
        min_games_first_percentage = np.min(games_finished_first_percentage)
        
        # Calculate statistics
        final_cumulative = cumulative_profits[:, -1]
        avg_cumulative = np.mean(final_cumulative)
        max_cumulative = np.max(final_cumulative)
        min_cumulative = np.min(final_cumulative)
        
        # Correctly calculate average, max, and min profit per game
        profit_per_game = final_cumulative / num_tournaments
        avg_profit_per_game = np.mean(profit_per_game)
        max_profit_per_game = np.max(profit_per_game)
        min_profit_per_game = np.min(profit_per_game)
        
        roi = final_cumulative / (buy_in * num_tournaments)
        avg_roi = np.mean(roi)
        max_roi = np.max(roi)
        min_roi = np.min(roi)
        
        avg_max_drawdown = np.mean(max_drawdowns)
        max_of_max_drawdowns = np.max(max_drawdowns)
        min_of_max_drawdowns = np.min(max_drawdowns)
        
        avg_max_drawdown_buyins = np.mean(max_drawdowns_buyins)
        max_of_max_drawdowns_buyins = np.max(max_drawdowns_buyins)
        min_of_max_drawdowns_buyins = np.min(max_drawdowns_buyins)
        
        avg_games_first = np.mean(games_finished_first)
        max_games_first = np.max(games_finished_first)
        min_games_first = np.min(games_finished_first)
        
        # **New Addition: Calculate Zero Payout Statistics**
        avg_zero_payout = np.mean(zero_payout_percentage)
        max_zero_payout = np.max(zero_payout_percentage)
        min_zero_payout = np.min(zero_payout_percentage)
    
    st.success("Simulation completed!")
    
    # Plotting
    st.subheader("Cumulative Profit/Loss Over Tournaments")
    # Added informational message about displayed simulations
    st.markdown("""
    **Note:** The graph below displays only 100 simulations for clarity. However, the Statistics Summary of Simulations and the distribution box plots consider all simulations.
    """)
    plt.figure(figsize=(10,6))
    # To avoid plotting too many lines, limit the number of simulations plotted
    max_lines = 100
    if num_simulations > max_lines:
        plot_indices = rng.choice(num_simulations, max_lines, replace=False)
    else:
        plot_indices = np.arange(num_simulations)
    for i in plot_indices:
        plt.plot(cumulative_profits[i], alpha=0.3)
    plt.xlabel("Number of Tournaments")
    plt.ylabel("Cumulative Profit ($)")
    plt.title("Monte Carlo Simulation of Cumulative Profit/Loss")
    plt.grid(True)
    st.pyplot(plt)
    
    # Statistics Summary
    st.subheader("Statistics Summary of Simulations")  # Renamed table
    
    # Added description under the title
    st.markdown("""
    The table below summarizes the key statistics from all the simulations.
    """)
    
    stats_data = {
        "Metric": [
            "Cumulative Profit/Loss ($)",
            "Average Profit per Tournament ($)",
            "ROI",
            "Max Drawdown ($)",
            "Max Drawdown (Buy-ins)",
            "Percentage of Tournaments Finished 1st",  # **Updated Metric**
            "Percentage of Tournaments with Zero Payout"
        ],
        "Average": [
            f"${avg_cumulative:,.2f}",
            f"${avg_profit_per_game:,.2f}",
            f"{avg_roi:.2%}",
            f"${avg_max_drawdown:,.2f}",
            f"{avg_max_drawdown_buyins:.2f} buy-ins",
            f"{avg_games_first_percentage:.2f}%",  # **Updated Value**
            f"{avg_zero_payout:.2f}%"  # **Existing Value**
        ],
        "Highest": [
            f"${max_cumulative:,.2f}",
            f"${max_profit_per_game:,.2f}",
            f"{max_roi:.2%}",
            f"${max_of_max_drawdowns:,.2f}",
            f"{max_of_max_drawdowns_buyins:.2f} buy-ins",
            f"{max_games_first_percentage:.2f}%",  # **Updated Value**
            f"{max_zero_payout:.2f}%"  # **Existing Value**
        ],
        "Lowest": [
            f"${min_cumulative:,.2f}",
            f"${min_profit_per_game:,.2f}",
            f"{min_roi:.2%}",
            f"${min_of_max_drawdowns:,.2f}",
            f"{min_of_max_drawdowns_buyins:.2f} buy-ins",
            f"{min_games_first_percentage:.2f}%",  # **Updated Value**
            f"{min_zero_payout:.2f}%"  # **Existing Value**
        ],
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.table(stats_df.set_index("Metric"))
    
    # Distribution Plots and Summary Tables
    st.subheader("Distribution Plots and Summary Tables")
    
    # Added description under the subheader
    st.markdown("""
    The box plots below illustrate the distribution of ROI and Max Draw-Down across all simulations. These visualizations help you understand the variability and potential risks associated with your tournament outcomes.
    """)
    
    # Create two columns for side-by-side box plots and their summaries
    box_col1, box_col2 = st.columns(2)
    
    with box_col1:
        # ROI Box Plot
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        bp1 = ax1.boxplot(roi, vert=True, patch_artist=True, showmeans=True,
                        boxprops=dict(facecolor="#AED6F1"),
                        medianprops=dict(color="blue"),
                        meanprops=dict(markerfacecolor="red", marker="D"))
        ax1.set_title("ROI Distribution")
        ax1.set_ylabel("ROI")  # Updated Y-axis label to indicate percentage
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0))  # Format Y-axis as percentage
        
        # Custom legend
        handles = [
            plt.Line2D([], [], color='blue', label='Median'),
            plt.Line2D([], [], marker='D', color='red', label='Mean', linestyle='None')
        ]
        ax1.legend(handles=handles, loc='upper right')
        
        st.pyplot(fig1)
        
        # Summary Table for ROI
        st.markdown("**ROI Distribution Summary**")
        
        # Convert roi to pandas Series
        roi_series = pd.Series(roi)
        roi_summary = roi_series.describe(percentiles=[0.25, 0.5, 0.75, 0.99])  # Included 99th percentile
        
        # Calculate outliers
        q1 = roi_summary['25%']
        q3 = roi_summary['75%']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        num_lower_outliers = np.sum(roi < lower_bound)
        num_upper_outliers = np.sum(roi > upper_bound)
        
        # Calculate percentage within whiskers
        num_within_whiskers = np.sum((roi >= lower_bound) & (roi <= upper_bound))
        percentage_within_whiskers = (num_within_whiskers / num_simulations) * 100
        
        # Calculate 99th percentile for ROI
        q99 = roi_series.quantile(0.99)
        
        # **New Addition: Calculate Percentage of Negative ROI**
        percentage_negative_roi = (roi < 0).mean() * 100  # Calculates the percentage
        
        roi_stats = {
            "Statistic": [
                "Mean",
                "Median",
                "25th Percentile (Q1)",
                "75th Percentile (Q3)",
                "99th Percentile",
                "Lower Whisker",
                "Upper Whisker",
                "Percentage of Simulations Within Whiskers",
                "Number of Lower Outliers",
                "Number of Upper Outliers",
                "Percentage of Simulations with Negative ROI"  # **Added Statistic**
            ],
            "Value": [
                f"{roi_summary['mean']:.2%}",  # Formatted as percentage
                f"{roi_summary['50%']:.2%}",
                f"{roi_summary['25%']:.2%}",
                f"{roi_summary['75%']:.2%}",
                f"{q99:.2%}",  # 99th percentile as percentage
                f"{lower_bound:.2%}",  # Formatted as percentage
                f"{upper_bound:.2%}",
                f"{percentage_within_whiskers:.2f}%",  # Fixed line
                int(num_lower_outliers),
                int(num_upper_outliers),
                f"{percentage_negative_roi:.2f}%"  # **Added Value**
            ]
        }
        
        roi_stats_df = pd.DataFrame(roi_stats)
        st.table(roi_stats_df.set_index("Statistic"))
    
    with box_col2:
        # Max Draw-Down (Buy-ins) Box Plot
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        bp2 = ax2.boxplot(max_drawdowns_buyins, vert=True, patch_artist=True, showmeans=True,
                         boxprops=dict(facecolor="#F9E79F"),
                         medianprops=dict(color="blue"),
                         meanprops=dict(markerfacecolor="red", marker="D"))
        ax2.set_title("Max Draw-Down (Buy-ins) Distribution")
        ax2.set_ylabel("Max Draw-Down (Buy-ins)")
        
        # Custom legend
        handles = [
            plt.Line2D([], [], color='blue', label='Median'),
            plt.Line2D([], [], marker='D', color='red', label='Mean', linestyle='None')
        ]
        ax2.legend(handles=handles, loc='upper right')
        
        st.pyplot(fig2)
        
        # Summary Table for Max Draw-Down (Buy-ins)
        st.markdown("**Max Draw-Down (Buy-ins) Distribution Summary**")
        
        # Convert max_drawdowns_buyins to pandas Series
        mdd_buyins_series = pd.Series(max_drawdowns_buyins)
        mdd_summary = mdd_buyins_series.describe(percentiles=[0.25, 0.5, 0.75, 0.99])  # Included 99th percentile
        
        # Calculate outliers
        q1_mdd = mdd_summary['25%']
        q3_mdd = mdd_summary['75%']
        iqr_mdd = q3_mdd - q1_mdd
        lower_bound_mdd = q1_mdd - 1.5 * iqr_mdd
        upper_bound_mdd = q3_mdd + 1.5 * iqr_mdd
        num_lower_outliers_mdd = np.sum(max_drawdowns_buyins < lower_bound_mdd)
        num_upper_outliers_mdd = np.sum(max_drawdowns_buyins > upper_bound_mdd)
        
        # Calculate percentage within whiskers
        num_within_whiskers_mdd = np.sum((max_drawdowns_buyins >= lower_bound_mdd) & (max_drawdowns_buyins <= upper_bound_mdd))
        percentage_within_whiskers_mdd = (num_within_whiskers_mdd / num_simulations) * 100
        
        # Calculate 99th percentile for Max Draw-Down (Buy-ins)
        q99_mdd = mdd_buyins_series.quantile(0.99)
        
        mdd_stats = {
            "Statistic": [
                "Mean",
                "Median",
                "25th Percentile (Q1)",
                "75th Percentile (Q3)",
                "99th Percentile",
                "Lower Whisker",
                "Upper Whisker",
                "Percentage of Simulations Within Whiskers",
                "Number of Lower Outliers",
                "Number of Upper Outliers"
            ],
            "Value": [
                f"{mdd_summary['mean']:.2f}",
                f"{mdd_summary['50%']:.2f}",
                f"{mdd_summary['25%']:.2f}",
                f"{mdd_summary['75%']:.2f}",
                f"{q99_mdd:.2f}",  # 99th percentile as float with two decimals
                f"{lower_bound_mdd:.2f}",  # Lower Whisker
                f"{upper_bound_mdd:.2f}",  # Upper Whisker
                f"{percentage_within_whiskers_mdd:.2f}%",  # Fixed line
                int(num_lower_outliers_mdd),
                int(num_upper_outliers_mdd)
            ]
        }
        
        mdd_stats_df = pd.DataFrame(mdd_stats)
        st.table(mdd_stats_df.set_index("Statistic"))
