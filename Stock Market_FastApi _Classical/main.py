import streamlit as st
import pandas as pd
from datetime import datetime
from agent import Agent
import json

def load_config():
    with open('config.json', 'r', encoding='utf-8') as file:
        return json.load(file)

def run_backtesting(config, start_date, end_date):
    agent = Agent(config)
    results_df = agent.backtesting(start_date, end_date, verbose=False)
    return results_df

def main():
    # Load configuration from JSON file
    config = load_config()

    # Streamlit App Layout
    st.title("Stock Price Prediction using LLMs")
    st.sidebar.header("User Input")

    # Inputs for Stock Symbol, Start Date, End Date
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol", config['stock_symbol'])
    start_date = st.sidebar.date_input("Start Date", datetime(2024, 9, 1))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 9, 30))

    # Run backtesting
    if st.sidebar.button("Run Backtesting"):
        st.write(f"Running backtesting for {stock_symbol} from {start_date} to {end_date}...")
        
        # Update config with user input
        config['stock_symbol'] = stock_symbol

        # Perform backtesting
        results_df = run_backtesting(config, start_date, end_date)
        
        # Display results
        st.subheader("Backtesting Results")
        st.write(results_df)

        # Save results to CSV
        results_df.to_csv('backtesting_results.csv', index=False)
        st.write("Results saved to 'backtesting_results.csv'.")

        # Plot the predicted vs actual prices
        st.subheader("Stock Price Prediction Plot")
        st.line_chart(results_df[['Date', 'Predicted Price']].set_index('Date'))
        st.line_chart(results_df[['Date', 'Actual Price']].set_index('Date'))

        # Evaluation Metrics
        st.subheader("Evaluation Metrics")
        agent = Agent(config)
        agent.evaluate_performance(results_df)

if __name__ == '__main__':
    main()