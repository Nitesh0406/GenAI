import openai
import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import plotly.graph_objects as go
from rapidfuzz import process
from src.utils.openai_api import get_supervisor_llm
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llm = get_supervisor_llm()


def DropPointCentralization_parameter_extraction_chain(llm=llm):

    # Prompt to instruct the model
    extract_parameters_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that extracts structured parameters to run a Drop Point Centralization Optimization algorithm from the user's request."),
        ("human", "{input}")
    ])

    # Matching schema with your algorithm's expected parameters
    extract_parameters_schema = {
        "name": "extract_DropPointCentralization_parameters",
        "description": "Extract parameters required for optimizing drop point centralization based on delivery data.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer": {
                    "type": "string",
                    "description": "The name of the customer for whom the optimization is being performed (e.g., 'TESCO STORES LTD')."
                },
                "start_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Start date for filtering delivery data (format: YYYY-MM-DD). Optional; if not provided, take 2023-01-01."
                },
                "end_date": {
                    "type": "string",
                    "format": "date",
                    "description": "End date for filtering delivery data (format: YYYY-MM-DD). Optional; if not provided, take 2025-02-28."
                },
                "no_of_drop_points": {
                    "type": "integer",
                    "description": "The number of drop points to include in the optimization. Default is always '1' if not specified (Important).",
                    "default": 1,
                    "minimum": 1
                },
                "rank_on": {
                    "type": "string",
                    "enum": ["Rate", "Volume", "Distance"],
                    "description": "Criteria used for ranking drop points. Must be one of: 'Rate', 'Volume', or 'Distance'. If not provided take Rate."
                }
            },
            "required": ["customer", "no_of_drop_points"]
        }
    }

    # Build the chain
    extract_parameters_chain = (
        extract_parameters_prompt
        | llm.bind(functions=[extract_parameters_schema], function_call={"name": "extract_DropPointCentralization_parameters"})
        | JsonOutputFunctionsParser()
    )

    return extract_parameters_chain



class DropPointCentralizationOptimizationAgent:
    def __init__(self, df: pd.DataFrame, amb_rc: pd.DataFrame, ambc_rc: pd.DataFrame, parameters: dict):
        
        self.df = df.copy()
        self.amb_rc = amb_rc
        self.ambc_rc = ambc_rc



        self.parameters = parameters
        self.customer = self.parameters.get("customer",None)
        self.start_date = self.parameters.get("start_date","2023-01-01")
        self.end_date = self.parameters.get("end_date","2025-02-28")
        self.no_of_drop_points = self.parameters.get("no_of_drop_points",1)
        self.rank_on = self.parameters.get("rank_on","Rate")
        self.customer_df = self.parameters.get("customer_df",None)
        self.new_drop_points = self.parameters.get("new_drop_points",None)
        

    def get_rate(self, postcode_prefix: str, num_pallets: int, rate_df: pd.DataFrame) -> float:
        try:
            if isinstance(num_pallets, int):
                if num_pallets == 1:
                    column_name = f"{num_pallets} Pallet"
                elif num_pallets <= 26:
                    column_name = f"{num_pallets} Pallets"
                else:
                    column_name = "Double Stacked Full Load"
            else:
                raise ValueError("Invalid input for number of pallets.")

            row = rate_df.loc[rate_df['Postcode Short'] == postcode_prefix.upper()]
            if row.empty or column_name not in rate_df.columns:
                return None

            rate_str = str(row.iloc[0][column_name])
            return float(rate_str.replace('£', '').replace(',', '').strip())
        except Exception as e:
            print(f"Error in get_rate: {e}")
            return None

    def prepare_data(self):
        
        self.found_customer = True
        df = self.df.dropna(subset=["ORDER_ID"])
        df = df[df['DELIVERED_QTY'] > 0]
        df['DISTANCE'] = df['DISTANCE'].fillna(df['DISTANCE'].mean()) # Filling `DISTANCE` NaNs with the mean values.
        if 'CUSTOMER_NAME' in df.columns:
            df.rename(columns={"CUSTOMER_NAME": "CUSTOMER_SEGMENT"}, inplace=True)

        filtered_df = df[df['DELIVERY_DATE'].between(self.start_date, self.end_date)]

        if self.customer in filtered_df['CUSTOMER_SEGMENT'].unique():
            customer_df = filtered_df[filtered_df['CUSTOMER_SEGMENT'] == self.customer]
            self.chosen_customer = self.customer
        elif self.customer in filtered_df['SHIP_TO_NAME'].unique():
            self.chosen_customer = filtered_df[filtered_df['SHIP_TO_NAME'] == self.customer]['CUSTOMER_SEGMENT'].unique()[0]
            customer_df = filtered_df[filtered_df['CUSTOMER_SEGMENT'] == self.chosen_customer]
        else:
            self.found_customer = False
            best_match, score, index = process.extractOne(self.customer, filtered_df['CUSTOMER_SEGMENT'].unique())
            self.chosen_customer = best_match
            customer_df = filtered_df[filtered_df['CUSTOMER_SEGMENT'] == best_match]
        

        customer_df = customer_df.groupby(
            ['SHIP_TO_NAME', 'SHORT_POSTCODE', 'POSTCODE']
        ).agg({
            'DELIVERED_QTY': 'sum',
            'DISTANCE': 'mean',
            'SALES': 'sum',
            'ORDER_ID': pd.Series.nunique,
            'PALLET_DISTRIBUTION': 'sum',
            'TRANSPORT_COST': 'sum'
        }).reset_index()

        customer_df['Average_Pallet_Per_Order'] = np.ceil(customer_df['PALLET_DISTRIBUTION'] / customer_df['ORDER_ID']).astype(int)
        customer_df['FULL_TRUCK_RATE'] = customer_df['SHORT_POSTCODE'].apply(lambda x: self.get_rate(x, 52, self.ambc_rc))

        self.customer_df = customer_df

    def rank_and_optimize(self):
        if self.rank_on == 'Volume':
            self.customer_df = self.customer_df.sort_values(by='PALLET_DISTRIBUTION', ascending=False)
        elif self.rank_on == 'Rate':
            self.customer_df = self.customer_df.sort_values(by='FULL_TRUCK_RATE', ascending=True)
        elif self.rank_on == 'Distance':
            self.customer_df = self.customer_df.sort_values(by='DISTANCE', ascending=True)

        df = self.customer_df.copy()
        total_orders = df['ORDER_ID'].sum()
        total_pallets = df['PALLET_DISTRIBUTION'].sum()

        if self.no_of_drop_points > len(df):
            raise ValueError("Number of drop points exceeds available data.")
        else:
            new_drops = df.iloc[:self.no_of_drop_points].copy()

        new_drops['Default_ratio_order'] = new_drops['ORDER_ID'] / new_drops['ORDER_ID'].sum()
        new_drops['New Orders Default'] = (new_drops['Default_ratio_order'] * total_orders).round().astype(int)

        new_drops['Default_ratio_pallet'] = new_drops['PALLET_DISTRIBUTION'] / new_drops['PALLET_DISTRIBUTION'].sum()
        new_drops['New Pallets Default'] = (new_drops['Default_ratio_pallet'] * total_pallets).round().astype(int)

        new_drops['New_Pallets_Per_Order'] = np.ceil(new_drops['New Pallets Default'] / new_drops['New Orders Default']).astype(int)

        new_drops['New Rate'] = new_drops.apply(lambda x: self.get_rate(x['SHORT_POSTCODE'], x['New_Pallets_Per_Order'], self.ambc_rc), axis=1)
        new_drops['New Transport Cost'] = new_drops['New Orders Default'] * new_drops['New Rate'] * new_drops['New_Pallets_Per_Order']

        self.new_drop_points = new_drops

    def get_summary_metrics(self):
        df = self.customer_df
        new_df = self.new_drop_points

        total_orders = df['ORDER_ID'].sum()
        total_pallets = df['PALLET_DISTRIBUTION'].sum()
        total_sales = df['SALES'].sum()
        total_cost = df['TRANSPORT_COST'].sum()
        c02_emissions = int(sum(df['DISTANCE'] * df['ORDER_ID'] * 2))
        no_of_drop_points = df['SHIP_TO_NAME'].unique()
        existing_km = df['DISTANCE'].sum()

        new_total_orders = new_df['New Orders Default'].sum()
        new_total_pallets = new_df['New Pallets Default'].sum()
        new_total_cost = new_df['New Transport Cost'].sum()
        new_c02_emissions = int(sum(new_df['DISTANCE'] * new_df['New Orders Default'] * 2))
        new_no_of_drop_points = new_df['SHIP_TO_NAME'].unique()
        new_existing_km = new_df['DISTANCE'].sum()

        self.existing_metric = {
            'Number of Drop Points': len(no_of_drop_points),
            'Total Transport Cost': round(total_cost, 2),
            'Per Pallet Cost': round(total_cost / total_pallets, 2),
            'CO2 Emissions (kg)': c02_emissions,
            'Average Distance per Shipment (km)': existing_km/total_orders,
            'Cost to Sales %': round((total_cost / total_sales) * 100, 2)
            
        }

        self.new_metric = {
            'Number of Drop Points': len(new_no_of_drop_points),
            'Total Transport Cost': round(new_total_cost, 2),
            'Per Pallet Cost': round(new_total_cost / new_total_pallets, 2),
            'CO2 Emissions (kg)': new_c02_emissions,
            'Average Distance per Shipment (km)': new_existing_km/total_orders,
            'Cost to Sales %': round((new_total_cost / total_sales) * 100, 2)
            
        }

        return self.existing_metric, self.new_metric

    def get_final_output_df(self):
        df = self.new_drop_points.copy()

        df = df[[
            'SHIP_TO_NAME', 'SHORT_POSTCODE', 'POSTCODE', 'ORDER_ID', 'PALLET_DISTRIBUTION',
            'TRANSPORT_COST', 'DISTANCE', 'FULL_TRUCK_RATE',
            'Default_ratio_order', 'New Orders Default', 'Default_ratio_pallet',
            'New Pallets Default', 'New_Pallets_Per_Order', 'New Rate',
            'New Transport Cost'
        ]]

        df.columns = [
            'Selected Drop Point', 'SHORT_POSTCODE', 'POSTCODE', 'Existing Total Orders',
            'Existing Total Pallets', 'Existing Transport Cost', 'DISTANCE', 'Full Truck Rate',
            'Order Distirbution %', 'New Total Orders', 'Pallet Distirbution %',
            'New Total Pallets', 'New Pallets Per Order', 'Rate Applied',
            'New Transport Cost'
        ]
        return df
    
    def get_cost_comparison_df(self):
        existing_cost = self.existing_metric['Total Transport Cost']
        new_cost = self.new_metric['Total Transport Cost']

        # Calculate percentage change
        percent_change = ((new_cost - existing_cost) / existing_cost) * 100
        change_label = f"{abs(percent_change):.1f}% {'increase' if percent_change > 0 else 'decrease'}"

        # Create bar chart
        fig = go.Figure()

        # Add bars
        fig.add_trace(go.Bar(
            x=["Existing", "Post Centralization"],
            y=[existing_cost, new_cost],
            marker=dict(color=["#1f77b4", "#2ca02c"]),  # Blue and green
            text=[f"£{existing_cost:,.0f}", f"£{new_cost:,.0f}"],
            textposition="outside",
            hovertemplate='Cost: £%{y:,.0f}<extra></extra>',
        ))

        # Add percentage change annotation between bars
        fig.add_annotation(
            x=0.5,
            y=max(existing_cost, new_cost) * 1.05,
            text=f"<b>{change_label}</b>",
            showarrow=False,
            font=dict(size=16, color="crimson" if percent_change > 0 else "green"),
        )

        # Layout styling
        fig.update_layout(
            title=dict(text="Transport Cost Comparison", x=0.5, font=dict(size=22)),
            yaxis_title="Cost (£)",
            yaxis=dict(showgrid=True, gridcolor='lightgrey'),
            xaxis=dict(title='', tickfont=dict(size=14)),
            plot_bgcolor='white',
            height=500,
            width=700,
            margin=dict(t=80, b=60),
        )
        return fig


    def generate_results(self,llm,parameters,question):
        self.parameters = parameters
        print("Parameters for Agent: ")
        for k,v in self.parameters.items():
            print(f"{k}:",v)
            
        self.prepare_data()
        self.rank_and_optimize()
        
        existing, new = self.get_summary_metrics()
        output_df = self.get_final_output_df()
        
        fig = self.get_cost_comparison_df()
        
        if self.found_customer:
            result = llm.invoke(
                f"""
                Based on the following inputs:
                - User question: {question}
                - Dictionary containing selected drop point, where all the drop points centralized to : {output_df.to_dict()}
                - Dictionary comparing existing vs. new scenarios across key KPIs: {existing,new}

                Please generate a **professional, concise summary in markdown format** that:
                - Directly answers the user’s question by synthesizing insights from all inputs
                - Identifies and highlights the optimal scenario
                - Clearly compares the existing and new scenarios across relevant KPIs
                - Summarizes all key extracted parameters in a single sentence
                - Maintains a clear, business-appropriate tone without redundant or irrelevant details, avoid salutations
                - Provide a summary of output_df containing selected drop point, where all the drop points centralized and also provide these column values POSTCODE, Order Distirbution %,Pallet Distirbution % 
                - All the cost should be in £K format
                """
            )
        else:
            result = llm.invoke(
                f"""
                Based on the following inputs:
                - User question: {question}
                - Dictionary containing selected drop point, where all the drop points centralized to : {output_df.to_dict()}
                - Dictionary comparing existing vs. new scenarios across key KPIs: {existing,new}
                - Name of Customer : {self.found_customer}

                Please generate a **professional, concise summary in markdown format** that:
                - Mention the name of customer we perofmed optimization on since exact match wasn't found: {self.found_customer}
                - Directly answers the user’s question by synthesizing insights from all inputs
                - Identifies and highlights the optimal scenario
                - Clearly compares the existing and new scenarios across relevant KPIs
                - Summarizes all key extracted parameters in a single sentence
                - Maintains a clear, business-appropriate tone without redundant or irrelevant details, avoid salutations
                - Provide a summary of output_df containing selected drop point, where all the drop points centralized and also provide these column values POSTCODE, Order Distirbution %,Pallet Distirbution % 
                - All the cost should be in £K format
                """
            )

        return {"final_response":result,"fig":[fig]}
