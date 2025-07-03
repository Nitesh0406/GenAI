## Setting up API key and environment related imports

from dotenv import load_dotenv, find_dotenv
from src.core.order_consolidation.static_consolidation import find_cost_savings
from src.core.order_consolidation.dynamic_consolidation import get_parameters_values,get_filtered_data
_ = load_dotenv(find_dotenv())

import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from langchain_core.messages import  AIMessage
from langchain.schema import HumanMessage


def compare_before_and_after_consolidation(original_df, consolidated_df,sales,capacity):
        """Compares shipments before and after consolidation."""
        # Workaround to calculate Number of Orders
        consolidated_df['Calculated_Orders'] = np.where(consolidated_df['Total Pallets'] <= capacity,1,np.ceil(consolidated_df['Total Pallets'] / capacity).astype(int))
        consolidated_df['CO2 Emission'] = consolidated_df['Distance'] * consolidated_df['Calculated_Orders'] * 2
        c02_emission_before = original_df['Distance'].sum() * 2
        c02_emission_after = consolidated_df['CO2 Emission'].sum()

        before = {
            "Days": original_df['SHIPPED_DATE'].nunique(),
            "Total Orders": len(original_df),
            "Pallets per Order": original_df['Total Pallets'].sum() / len(original_df),
            "Total Transport Cost": round(original_df['shipment_cost'].sum(),2),
            "Cost per pallet": round(original_df['shipment_cost'].sum() / original_df['Total Pallets'].sum(), 2),
            'Cost to Sales Ratio': (original_df['shipment_cost'].sum() / sales) * 100,
            'CO2 Emission (kg)': round(c02_emission_before, 1)
        }

        after = {
            "Days": consolidated_df['Date'].nunique(),
            "Total Orders": consolidated_df['Calculated_Orders'].sum(),
            "Pallets per Order": consolidated_df['Total Pallets'].sum() / consolidated_df['Calculated_Orders'].sum(),
            "Total Transport Cost": round(consolidated_df['consolidated_shipment_cost'].sum(), 2),
            "Cost per pallet": round(consolidated_df['consolidated_shipment_cost'].sum() / consolidated_df['Total Pallets'].sum(), 2),
            'Cost to Sales Ratio': (consolidated_df['consolidated_shipment_cost'].sum() / sales) * 100,
            'CO2 Emission (kg)': round(c02_emission_after, 1)
        }

        percentage_change = {
            key: round(((after[key] - before[key]) / before[key]) * 100, 2) for key in before
        }

        comparison_df = pd.DataFrame({"Before": before, "After": after, "% Change": percentage_change})
        return comparison_df

def calculate_metrics(original_df, consolidated_df,total_sales,capacity):

    # Workaround to calculate Number of Orders
    consolidated_df['Calculated_Orders'] = np.where(consolidated_df['Total Pallets'] <= capacity,1,np.ceil(consolidated_df['Total Pallets'] / capacity).astype(int))

    total_shipments = consolidated_df['Calculated_Orders'].sum()
    total_pallets = consolidated_df['Total Pallets'].sum()
    original_cost = original_df['shipment_cost'].sum()
    consolidated_cost = consolidated_df['consolidated_shipment_cost'].sum()
    cost_savings = round(original_cost-consolidated_cost,2)
    percent_savings = (cost_savings / original_cost) * 100
    original_shipment_days = len(original_df['Date'].unique())
    consolidated_shipment_days = len(consolidated_df['Date'].unique())
    cost_to_sales_ratio_baseline = (original_cost / total_sales) * 100
    cost_to_sales_ratio_after_consolidation = (consolidated_cost / total_sales) * 100

    print(consolidated_df.columns)

    metrics = {
        'Total Orders': len(original_df),
        'Total Shipments': total_shipments,
        'Total Pallets': total_pallets,
        'Total Shipment Cost': consolidated_cost,
        'Total Baseline Cost': original_cost,
        'Cost Savings': round(cost_savings, 1),
        'Percent Savings': percent_savings,
        # 'CO2 Emission (kg)': round(consolidated_df['Distance'].sum() * 2, 1),
    }
    return metrics

def get_static_savings(llm,question,shipment_df,rate_card,state_parameters):
    chat_history = [{"Human": question}]
    extracted_params = {k: v for k, v in state_parameters.items() if k != "enriched_query"}

    chat_history.append({"Agent": f"Parameters extracted: {extracted_params}"})
    capacity = extracted_params['total_shipment_capacity']
    cost_saving_results = find_cost_savings(shipment_df,rate_card,extracted_params)
    
    consolidated_df = cost_saving_results['consolidated_data']
    consolidated_df['Date']=consolidated_df['UPDATED_DATE']
    original_df = cost_saving_results['aggregated_data']
    original_df['Date']=original_df['SHIPPED_DATE']


    filter_data = get_filtered_data(state_parameters,shipment_df)
    metrics = calculate_metrics(original_df, consolidated_df, filter_data['SALES'].sum(),capacity)
    best_scenario = cost_saving_results['best_scenario']
    best_scenario.update(metrics)
    best_scenario.pop('num_shipments', None)  # Remove num_shipments if it exists
    chat_history.append({"Agent": f"Scenarios of all possible days: {cost_saving_results['all_results']}"})
    chat_history.append({"Agent": f"Best scenarios for cost savings: {best_scenario}"})

    capacity = extracted_params['total_shipment_capacity']
    comparison_results = compare_before_and_after_consolidation(original_df, consolidated_df,filter_data['SALES'].sum(),capacity)
    chat_history.append({"Agent": f"Comparison results: {comparison_results.to_dict()}"})


    chat = []
    for msg in chat_history:
        key, value = list(msg.items())[0]
        if "Agent" in key:
            if type(value) is not str:
                value = str(value)
            chat.append(AIMessage(content=value))
        else:
            chat.append(HumanMessage(content=value))

    result = llm.invoke(
        f"""This is the response provided by the Static Cost Optimization Agent: {chat}. 
        Generate a final response to be shown to the user. 
        - Show best scenario results for cost savings.
        - Show comparison results in a tabular format clearly compares the existing and new scenarios across relevant KPIs.
        - List all extracted parameters separately and show in a single sentence.
        - Keep the tone professional and clear, avoid salutations and generate outpute in a factual manner
        - All the cost should be in £K format
        """)
    
    cost_saving_results['final_response'] = f"""
    {result.content}  
    """
    return cost_saving_results








