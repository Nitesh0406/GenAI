## Setting up API key and environment related imports

import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import HumanMessage
from src.core.order_consolidation.dynamic_consolidation import (get_filtered_data,
                                                                get_parameters_values,
                                                                consolidate_shipments,
                                                                calculate_metrics,
                                                                analyze_consolidation_distribution, agent_wrapper)
_ = load_dotenv(find_dotenv())
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


class AgenticCostOptimizer:
    def __init__(self, llm, parameters):
        """
        Initialize the Agentic Cost Optimizer.

        :param llm: The LLM model to use for queries.
        :param parameters: Dictionary containing required parameters.
        """

        self.llm = llm
        self.parameters = parameters
        self.df = parameters.get("df", pd.DataFrame())


    def load_data(self):
        complete_input = os.path.join(os.getcwd() , "src/data/Complete Input.xlsx")
        rate_card_ambient = pd.read_excel(complete_input, sheet_name='AMBIENT')
        rate_card_ambcontrol = pd.read_excel(complete_input, sheet_name='AMBCONTROL')
        return {"rate_card_ambient": rate_card_ambient, "rate_card_ambcontrol": rate_card_ambcontrol}


    def get_filtered_df_from_question(self):
        """Extracts filtered data based on user query parameters."""
        group_field = 'SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME'
        df = self.parameters['df']
        df['SHIPPED_DATE'] = pd.to_datetime(df['SHIPPED_DATE'], dayfirst=True)

        df = get_filtered_data(self.parameters, df)
        if df.empty:
            raise ValueError("No data available for selected parameters. Try again!")
        return df

    def get_cost_saving_data(self):
        """Runs cost-saving algorithm and returns result DataFrame."""
        
        df = self.get_filtered_df_from_question()
        group_field = 'SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME'
        
        df['GROUP'] = df['SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME']
        grouped = df.groupby(['PROD TYPE', 'GROUP'])
        date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])

        
        best_metrics=None
        best_consolidated_shipments=None
        best_params = None

        all_results = []
        rate_card = self.load_data()
        for shipment_window in range(self.parameters["shipment_window_range"][0], self.parameters["shipment_window_range"][1] + 1):
            high_priority_limit = 0
            all_consolidated_shipments = []
            for _, group_df in grouped:
                consolidated_shipments, _ = consolidate_shipments(
                    group_df, high_priority_limit, self.parameters["utilization_threshold"], shipment_window, date_range, lambda: None, self.parameters["total_shipment_capacity"],rate_card
                )
                all_consolidated_shipments.extend(consolidated_shipments)
            
            metrics = calculate_metrics(all_consolidated_shipments, df)
            distribution, distribution_percentage = analyze_consolidation_distribution(all_consolidated_shipments, df)
            
            result = {
                'Shipment Window': shipment_window,
                'Total Orders': metrics['Total Orders'],
                'Total Shipments': metrics['Total Shipments'],
                'Total Shipment Cost': round(metrics['Total Shipment Cost'], 1),
                'Total Baseline Cost': round(metrics['Total Baseline Cost'], 1),
                'Cost Savings': metrics['Cost Savings'],
                'Percent Savings': round(metrics['Percent Savings'], 1),
                'Average Utilization': round(metrics['Average Utilization'], 1),
                # 'CO2 Emission (kg)': round(metrics['CO2 Emission (kg)'], 1),
            }
            all_results.append(result)

        # Update best results if current combination is better
        if best_metrics is None or metrics['Cost Savings'] > best_metrics['Cost Savings']:
            best_metrics = metrics
            best_consolidated_shipments = all_consolidated_shipments
            best_params = (shipment_window, high_priority_limit, self.parameters["utilization_threshold"])

        # Updating the parameters with adding shipment window vs cost saving table..    
        self.parameters['all_results'] = pd.DataFrame(all_results)
        self.parameters['best_params'] = best_params

    def consolidate_for_shipment_window(self):
        """Runs consolidation algorithm based on the selected shipment window."""
        df = self.get_filtered_df_from_question()
        df['GROUP'] = df['SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME']
        grouped = df.groupby(['PROD TYPE', 'GROUP'])
        date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])

        rate_card = self.load_data()
        all_consolidated_shipments = []
        for _, group_df in grouped:
            consolidated_shipments, _ = consolidate_shipments(
                group_df, 0, 95, self.parameters['window'], date_range, lambda: None, self.parameters["total_shipment_capacity"],
                rate_card
            )
            all_consolidated_shipments.extend(consolidated_shipments)

        selected_postcodes = ", ".join(self.parameters["selected_postcodes"]) if self.parameters[
            "selected_postcodes"] else "All Postcodes"
        selected_customers = ", ".join(self.parameters["selected_customers"]) if self.parameters[
            "selected_customers"] else "All Customers"

        metrics = calculate_metrics(all_consolidated_shipments, df)

        self.parameters['all_consolidated_shipments'] = pd.DataFrame(all_consolidated_shipments)
        self.parameters['metrics'] = metrics
        self.parameters['filtered_df'] = df

    def compare_before_and_after_consolidation(self):
        """Compares shipments before and after consolidation."""
        consolidated_df = self.parameters['all_consolidated_shipments']
        df = self.get_filtered_df_from_question()
        
        total_sales = df['SALES'].sum()
        print("Total Sales:", total_sales,int(total_sales))
        total_shipment_cost = int(consolidated_df['Shipment Cost'].sum())
        total_baseline_cost = int(consolidated_df['Baseline Cost'].sum())

        co2_emission_before = df['Distance'].sum() * 2
        distance_df = df.groupby(['SHORT_POSTCODE'])['Distance'].mean().reset_index()
        consolidated_df = consolidated_df.merge(distance_df, on='SHORT_POSTCODE', how='left')
        consolidated_df['CO2 Emission (kg)'] = consolidated_df['Distance'] * 2
        co2_emission_after = consolidated_df['CO2 Emission (kg)'].sum()
        
        before = {
            "Days": df['SHIPPED_DATE'].nunique(),
            "Total Orders": len(df),
            "Pallets per Order": df['Total Pallets'].sum() / len(df),
            "Total Transport Cost": total_baseline_cost,
            "Cost per pallet": round(total_baseline_cost / df['Total Pallets'].sum(),2),
            'Cost to Sales Ratio': (total_baseline_cost/total_sales)*100,
            "CO2 Emission (kg)": round(co2_emission_before, 1)
        }
        after = {
            "Days": consolidated_df['Date'].nunique(),
            "Total Orders": len(consolidated_df),
            "Pallets per Order": consolidated_df['Total Pallets'].sum() / len(consolidated_df),
            "Total Transport Cost": total_shipment_cost,
            "Cost per pallet": round(total_shipment_cost / df['Total Pallets'].sum(),2),
            'Cost to Sales Ratio': (total_shipment_cost/total_sales)*100,
            "CO2 Emission (kg)": round(co2_emission_after, 1)
        }
        print("Before consolidation metrics:", before)
        print("After consolidation metrics:", after)
        percentage_change = {
            key: round(((after[key] - before[key]) / before[key]) * 100, 2) for key in before
        }

        comparison_df = pd.DataFrame({"Before": before, "After": after, "% Change": percentage_change})
        self.parameters["comparison_df"] = comparison_df


    def run_agent_query(self,agent, query, dataframe, max_attempts=3):
        """Runs an agent query with up to `max_attempts` retries on failure.

        Args:
            agent: The agent to invoke.
            query (str): The query to pass to the agent.
            dataframe (pd.DataFrame): DataFrame for response context.
            max_attempts (int, optional): Maximum retry attempts. Defaults to 3.

        Returns:
            str: Final answer or error message after attempts.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                response = agent.invoke(query)
                response_ = agent_wrapper(response, dataframe)
                return response_["final_answer"]

            except Exception as e:
                attempt += 1
                if attempt == max_attempts:
                    return f"Error: {e}"
                
    def handle_query(self, state_parameters):
        """Handles user queries dynamically with conversation history and data processing."""
        question = state_parameters["enriched_query"]
        chat_history = []
        chat_history.append({"Human": question})

        # Extract parameters from question
        extracted_params  = {k: v for k, v in state_parameters.items() if k != "enriched_query"}        
        self.parameters.update(extracted_params)
        chat_history.append({"Agent": f"Parameters extracted: {extracted_params}"})

        self.get_cost_saving_data()
        # Identify row with maximum cost savings
        max_savings_row = self.parameters['all_results'].loc[
            self.parameters['all_results']['Cost Savings'].idxmax()
        ].to_dict()
        chat_history.append({"Agent": f"Optimum results: {max_savings_row}"})
        user_window = None  # Replace with user input logic if needed
        self.parameters["window"] = int(user_window) if user_window else max_savings_row['Shipment Window']
        self.consolidate_for_shipment_window()
        self.compare_before_and_after_consolidation()
        comparison_results = self.parameters["comparison_df"].to_dict()
        chat_history.append({"Agent": f"Comparison results: {comparison_results}"})
        chat = []
        for msg in chat_history:
            key, value = list(msg.items())[0]
            if "Agent" in key:
                if type(value) is not str:
                    value = str(value)
                chat.append(AIMessage(content=value))
            else:
                chat.append(HumanMessage(content=value))

        result = self.llm.invoke(
                f"""This is the response provided by the Dynamic Cost Optimization Agent: {chat}. 
                Generate a final response to be shown to the user. 
                - Show optimum results.
                - Show comparison results in a tabular format clearly compares the existing and new scenarios across relevant KPIs.
                - List all extracted parameters separately and show in a single sentence.
                - Keep the tone professional and clear, avoid salutations and generate outpute in a factual manner
                - All the cost should be in £K format
                """
            )

        # Structuring the final response
        self.parameters['final_response'] = f"""
        {result.content}
        """
        print("Reponse END from Dynamic Agent")
        return self.parameters

