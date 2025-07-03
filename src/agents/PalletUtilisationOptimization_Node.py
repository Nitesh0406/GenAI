import openai
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import plotly.express as px
import plotly.graph_objects as go
import warnings
from dotenv import load_dotenv, find_dotenv
from src.utils.openai_api import get_supervisor_llm

warnings.filterwarnings("ignore")
_ = load_dotenv(find_dotenv())
llm = get_supervisor_llm()

def PalletUtilisation_parameter_extraction_chain(llm=llm):

    # Defining role prompt for our LLM
    extract_pallet_params_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that extracts structured parameters from user questions."),
        ("human", "{input}")
    ])

    # Updated function schema with all parameters
    extract_pallet_params_schema = {
        "name": "extract_pallet_optimization_parameters",
        "description": "Extract all required parameters to run pallet utilization optimization from the user's query.",
        "parameters": {
            "type": "object",
            "properties": {
                "mat": {
                    "type": "string",
                    "description": "The SKU or material number mentioned in the user query  (e.g., '5000033129')."
                },
                "pallet_type": {
                    "type": "string",
                    "enum": ["IND", "EURO","EURO CCG1"],
                    "description": "The type of pallet: 'IND', 'EURO' or 'EURO CCG1'. EURO in Germany should be EURO CCG1.If not mentioned, default to 'IND'."
                },
                "inbound_transport_rate": {
                    "type": "number",
                    "description": "The rate for inbound transportation (e.g., 23.08)."
                },
                "inbound_handling_rate": {
                    "type": "number",
                    "description": "The rate for inbound handling (e.g., 5.23)."
                },
                "storage_rate": {
                    "type": "number",
                    "description": "The rate for storage (e.g., 11.80)."
                },
                "outbound_handling_rate": {
                    "type": "number",
                    "description": "The rate for outbound handling (e.g., 2.50)."
                },
                "outbound_transport_rate": {
                    "type": "number",
                    "description": "The rate for outbound transportation (e.g., 30)."
                },
                "cost_per_wooden_pallet": {
                    "type": "number",
                    "description": "The cost per wooden pallet (e.g., 7)."
                },
                "other": {
                    "type": "number",
                    "description": "Other miscellaneous costs (e.g., 0)."
                },
                "double_stack_on_storage": {
                    "type": "boolean",
                    "description": "Keep default as True ,Whether double stacking is enabled during storage (true/false). User can mention whether double stacking is allowed in storage/warehouse or not based on it change this variable."
                },
                "ocean_freight": {
                    "type": "boolean",
                    "description": "Keep default as False, if mentioned by the user extract accordingly, user can mention whether ibound is via ocean freight or not also if container inbound is mentioned keep this true."
                }
                
            },
            "required": ["mat"]
        }
    }

    # Creating a parameter extraction chain
    extract_parameters_chain = (
        extract_pallet_params_prompt
        | llm.bind(functions=[extract_pallet_params_schema], function_call={"name": "extract_pallet_optimization_parameters"})
        | JsonOutputFunctionsParser()
    )

    return extract_parameters_chain


class PalletUtilisationOptimisationAgent:
    def __init__(self,parameters):
        self.parameters = parameters
        self.parameters["pallet_type"] = parameters.get("pallet_type","IND")
        self.parameters["inbound_transport_rate"] = parameters.get("inbound_transport_rate",23.08)
        self.parameters["inbound_handling_rate"] = parameters.get("inbound_handling_rate",5.23)
        self.parameters["storage_rate"] = parameters.get("storage_rate",11.80)
        self.parameters["outbound_handling_rate"] = parameters.get("outbound_handling_rate",2.50)
        self.parameters["outbound_transport_rate"] = parameters.get("outbound_transport_rate",30)
        self.parameters["cost_per_wooden_pallet"] = parameters.get("cost_per_wooden_pallet",7)
        self.parameters["other"] = parameters.get("other",0)
        # self.double_stack = double_stack
        self.parameters["double_stack_on_storage"] = parameters.get("double_stack_on_storage",True)
        self.parameters["ocean_freight"] = parameters.get("ocean_freight",False)


    def calculate_pallet_config(self):
        df = self._mat_df
        stick_per_shipper = df['ST/SHP'].values[0]
        shipper_per_layer = df['SHP/LYR'].values[0]
        old_stick_per_pallet = df['ST/PLT'].values[0]
        total_volume = df['Volume'].values[0]
        existing_pallet_required = int(total_volume // old_stick_per_pallet)
        cost_breakdown_dict = {}

        shipper_len = df['SHIPPER_LENGTH (mm)'].values[0] / 1000
        shipper_wid = df['SHIPPER_WIDTH (mm)'].values[0] / 1000
        shipper_box_height = df['SHIPPER_HEIGHT (mm)'].values[0] / 1000
        weight_shipper = df['SHIPPER_WT'].values[0]
        existing_layer = df['LYR/PLT'].values[0]
        existing_height = df['PLT_HEIGHT (mm)'].values[0] / 1000
        stick_per_shipper = df['ST/SHP'].values[0]

        if self.pallet_type == 'IND':
            wid_pallet = 1.0
            len_pallet = 1.2
            height_params = [1.25, 1.40, 1.65, 1.85]
            double_stacking_height = 1.25
        elif self.pallet_type == 'EURO':
            wid_pallet = 0.8
            len_pallet = 1.2
            height_params = [1.25, 1.40, 1.65, 1.85]
            # height_param_ccg1 = []
            double_stacking_height = 1.25
        elif self.pallet_type == 'EURO CCG1':
            wid_pallet = 0.8
            len_pallet = 1.2
            height_params = [1.05, 1.40, 1.65, 1.95]
            # height_param_ccg1 = []
            double_stacking_height = 1.05
        else:
            raise ValueError("Invalid pallet type. Use 'IND','EURO' or 'EURO CCG1'")

        layer_area = wid_pallet * len_pallet
        shipper_area = shipper_len * shipper_wid
        shipper_volume = shipper_area * shipper_box_height
        old_util = (shipper_area * shipper_per_layer) / layer_area * 100
        new_shipper_per_layer = shipper_per_layer

        while True:
            next_util = (shipper_area * (new_shipper_per_layer + 1)) / layer_area * 100
            if next_util > 100 or (new_shipper_per_layer+1)*existing_layer*weight_shipper > 500:
                break
            new_shipper_per_layer += 1

        new_util = (shipper_area * new_shipper_per_layer) / layer_area * 100

        height = {}
        units_per_pallet = {}
        shipper_boxes_per_pallet = {}
        total_pal = {}
        volumetric_efficiency = {}
        layer_efficiency = {}
        total_cost = {}

        result = {
            "SKU": self.mat,
            "Old Shipper per Layer": shipper_per_layer,
            "Old Layer Utilization (%)": round(old_util, 2),
            "New Shipper per Layer": new_shipper_per_layer,
            "New Layer Utilization (%)": round(new_util, 2),
            "Existing Required Pallet": existing_pallet_required,
            "Required Pallet for Optimized Layer": int(total_volume / (existing_layer * new_shipper_per_layer * stick_per_shipper)),
            "Weight for Optimized Layer": weight_shipper*new_shipper_per_layer*existing_layer,
            "Existing Cost" : self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=existing_pallet_required, double_stack=False),
            "Cost at Optimized Layer": self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=int(total_volume / (existing_layer * new_shipper_per_layer * stick_per_shipper)), double_stack=False),
        }

        height['Existing Height'] = existing_height
        height['Existing Height - Optimized Layer'] = existing_height
        units_per_pallet['Existing Height'] = existing_layer * shipper_per_layer * stick_per_shipper
        units_per_pallet['Existing Height - Optimized Layer'] = existing_layer * new_shipper_per_layer * stick_per_shipper
        shipper_boxes_per_pallet['Existing Height'] = existing_layer * shipper_per_layer
        shipper_boxes_per_pallet['Existing Height - Optimized Layer'] = existing_layer * new_shipper_per_layer
        total_pal['Existing Height'] = existing_pallet_required
        total_pal['Existing Height - Optimized Layer'] = int(total_volume / (existing_layer * new_shipper_per_layer * stick_per_shipper))
        volumetric_efficiency['Existing Height'] = (shipper_volume * shipper_per_layer * existing_layer) / (wid_pallet * len_pallet * existing_height) * 100
        volumetric_efficiency['Existing Height - Optimized Layer'] = (shipper_volume * new_shipper_per_layer * existing_layer) / (wid_pallet * len_pallet * existing_height) * 100
        layer_efficiency['Existing Height'] = (shipper_area * shipper_per_layer) / layer_area * 100
        layer_efficiency['Existing Height - Optimized Layer'] = (shipper_area * new_shipper_per_layer) / layer_area * 100
        total_cost['Existing Height'] = result['Existing Cost']
        total_cost['Existing Height - Optimized Layer'] = result['Cost at Optimized Layer']



        cost_breakdown_dict['Existing Cost'] = self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=existing_pallet_required, double_stack=False, return_breakdown=True)
        cost_breakdown_dict['Cost at Optimized Layer'] = self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=int(total_volume / (existing_layer * new_shipper_per_layer * stick_per_shipper)), double_stack=False, return_breakdown=True)

        for height_param in height_params:

            new_layer = existing_layer
            new_height = existing_height
            if existing_height > height_param:
                while new_height > height_param:
                    new_height -= shipper_box_height
                    new_layer -= 1
            else:
                while new_height + shipper_box_height <= height_param and new_layer*new_shipper_per_layer*weight_shipper<500:
                # while new_height + shipper_box_height <= height_param:
                    # print(new_layer*new_shipper_per_layer*weight_shipper)
                    new_height += shipper_box_height
                    new_layer += 1


            new_stick_per_pallet = int(new_shipper_per_layer * new_layer) * stick_per_shipper
            result[f'New ST/PLT at {height_param}m with Optimized Layer'] = new_stick_per_pallet
            result[f'Pallet Required at {height_param}m with Optimized Layer'] = int(total_volume / new_stick_per_pallet)
            result[f'Weight at {height_param}m with Optimized Layer'] = weight_shipper * new_shipper_per_layer * new_layer
            result[f'Cost at {height_param}m with Optimized Layer'] = self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=result[f'Pallet Required at {height_param}m with Optimized Layer'],double_stack=False,return_breakdown=False)
            
            # ----- TABLE VIEW -----
            height[f'{height_param}m with Optimized Layer'] = new_height
            units_per_pallet[f'{height_param}m with Optimized Layer'] = new_layer * new_shipper_per_layer * stick_per_shipper
            shipper_boxes_per_pallet[f'{height_param}m with Optimized Layer'] = new_layer * new_shipper_per_layer
            total_pal[f'{height_param}m with Optimized Layer'] = int(total_volume / new_stick_per_pallet)
            volumetric_efficiency[f'{height_param}m with Optimized Layer'] = (shipper_volume * new_shipper_per_layer * new_layer) / (wid_pallet * len_pallet * new_height) * 100
            layer_efficiency[f'{height_param}m with Optimized Layer'] = (shipper_area * new_shipper_per_layer) / layer_area * 100
            total_cost[f'{height_param}m with Optimized Layer'] = result[f'Cost at {height_param}m with Optimized Layer']
            # ----------------------

            existing_stick_per_pallet = int(shipper_per_layer * new_layer) * stick_per_shipper
            result[f'New ST/PLT at {height_param}m with Existing Layer'] = existing_stick_per_pallet
            result[f'Pallet Required at {height_param}m with Existing Layer'] = int(total_volume / existing_stick_per_pallet)
            result[f'Weight at {height_param}m with Existing Layer'] = weight_shipper * shipper_per_layer * new_layer
            result[f'Cost at {height_param}m with Existing Layer'] = self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=result[f'Pallet Required at {height_param}m with Existing Layer'],double_stack=False,return_breakdown=False)
            
            cost_breakdown_dict[f'Cost at {height_param}m with Optimized Layer'] = self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=result[f'Pallet Required at {height_param}m with Optimized Layer'],double_stack=False,return_breakdown=True)
            cost_breakdown_dict[f'Cost at {height_param}m with Existing Layer'] = self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=result[f'Pallet Required at {height_param}m with Existing Layer'],double_stack=False,return_breakdown=True)
            
            # ----- TABLE VIEW -----
            height[f'{height_param}m with Existing Layer'] = new_height
            units_per_pallet[f'{height_param}m with Existing Layer'] = new_layer * shipper_per_layer * stick_per_shipper
            shipper_boxes_per_pallet[f'{height_param}m with Existing Layer'] = new_layer * shipper_per_layer
            total_pal[f'{height_param}m with Existing Layer'] = int(total_volume / existing_stick_per_pallet)
            volumetric_efficiency[f'{height_param}m with Existing Layer'] = (shipper_volume * shipper_per_layer * new_layer) / (wid_pallet * len_pallet * new_height) * 100
            layer_efficiency[f'{height_param}m with Existing Layer'] = (shipper_area * shipper_per_layer) / layer_area * 100
            total_cost[f'{height_param}m with Existing Layer'] = result[f'Cost at {height_param}m with Existing Layer']
            # ----------------------

            if height_param <= double_stacking_height:
                result[f'Pallet Required at {height_param}m Double Stacked with Optimized Layer'] = result[f'Pallet Required at {height_param}m with Optimized Layer'] / 2
                result[f'Pallet Required at {height_param}m Double Stacked with Existing Layer'] = result[f'Pallet Required at {height_param}m with Existing Layer'] / 2
                result[f'Cost at {height_param}m Double Stacked with Existing Layer'] = self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=result[f'Pallet Required at {height_param}m Double Stacked with Existing Layer'],double_stack=True,return_breakdown=False)
                result[f'Cost at {height_param}m Double Stacked with Optimized Layer'] = self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=result[f'Pallet Required at {height_param}m Double Stacked with Optimized Layer'],double_stack=True,return_breakdown=False)
                cost_breakdown_dict[f'Cost at {height_param}m Double Stacked with Optimized Layer'] = self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=result[f'Pallet Required at {height_param}m Double Stacked with Optimized Layer'],double_stack=True,return_breakdown=True)
                cost_breakdown_dict[f'Cost at {height_param}m Double Stacked with Existing Layer'] = self.calculate_cost(existing_pallet=existing_pallet_required,total_pallets=result[f'Pallet Required at {height_param}m Double Stacked with Existing Layer'],double_stack=True,return_breakdown=True)

                # ----- TABLE VIEW -----
                height[f'{height_param}m Double Stacked with Optimized Layer'] = new_height
                units_per_pallet[f'{height_param}m Double Stacked with Optimized Layer'] = new_layer * new_shipper_per_layer * stick_per_shipper * 2
                shipper_boxes_per_pallet[f'{height_param}m Double Stacked with Optimized Layer'] = new_layer * new_shipper_per_layer
                total_pal[f'{height_param}m Double Stacked with Optimized Layer'] = int(total_volume / new_stick_per_pallet) / 2
                volumetric_efficiency[f'{height_param}m Double Stacked with Optimized Layer'] = (shipper_volume * new_shipper_per_layer * new_layer) / (wid_pallet * len_pallet * new_height) * 100
                layer_efficiency[f'{height_param}m Double Stacked with Optimized Layer'] = (shipper_area * new_shipper_per_layer) / layer_area * 100
                total_cost[f'{height_param}m Double Stacked with Optimized Layer'] = result[f'Cost at {height_param}m Double Stacked with Optimized Layer']
                # ----------------------

                # ----- TABLE VIEW -----
                height[f'{height_param}m Double Stacked with Existing Layer'] = new_height
                units_per_pallet[f'{height_param}m Double Stacked with Existing Layer'] = new_layer * shipper_per_layer * stick_per_shipper * 2
                shipper_boxes_per_pallet[f'{height_param}m Double Stacked with Existing Layer'] = new_layer * shipper_per_layer
                total_pal[f'{height_param}m Double Stacked with Existing Layer'] = int(total_volume / existing_stick_per_pallet) / 2
                volumetric_efficiency[f'{height_param}m Double Stacked with Existing Layer'] = (shipper_volume * shipper_per_layer * new_layer) / (wid_pallet * len_pallet * new_height) * 100
                layer_efficiency[f'{height_param}m Double Stacked with Existing Layer'] = (shipper_area * shipper_per_layer) / layer_area * 100
                total_cost[f'{height_param}m Double Stacked with Existing Layer'] = result[f'Cost at {height_param}m Double Stacked with Existing Layer']
                # ----------------------
        
        table_dict = {'Height (m)': height,
                      'Units per Pallet': units_per_pallet,
                      'Shipper Boxes per Pallet': shipper_boxes_per_pallet,
                      'Total Pallets Required': total_pal,
                      'Volumetric Efficiency (%)': volumetric_efficiency,
                      'Layer Efficiency (%)': layer_efficiency,
                      'Total Cost (£)': total_cost}
        table_df = pd.DataFrame.from_dict(table_dict, orient='index')
        optimization_summary = pd.DataFrame([result],columns=result.keys())
        cost_breakdown_df = pd.DataFrame.from_dict(cost_breakdown_dict, orient='index')
        return optimization_summary, cost_breakdown_df, table_df, table_dict

    def calculate_cost(self,existing_pallet,total_pallets, double_stack, return_breakdown=False):

        if self.parameters["ocean_freight"]:
            inbound_transport_rate = self.parameters["inbound_transport_rate"] * existing_pallet
            self.parameters["inbound_handling_rate"] = self.parameters["inbound_handling_rate"] + 4.80 
        else:
            inbound_transport_rate = self.parameters["inbound_transport_rate"] * total_pallets

        
        if double_stack:
            if self.parameters["double_stack_on_storage"]:
                components = {
                    'Inbound Transport': inbound_transport_rate,
                    'Inbound Handling': total_pallets * 2 * self.parameters["inbound_handling_rate"],
                    'Storage': total_pallets * self.parameters["storage_rate"],
                    'Outbound Handling': total_pallets * self.parameters["outbound_handling_rate"],
                    'Outbound Transport': total_pallets * self.parameters["outbound_transport_rate"],
                    'Wooden Pallet': total_pallets * 2 * self.parameters["cost_per_wooden_pallet"],
                    'Other': total_pallets * self.parameters["other"],
                }
            else:
                components = {
                    'Inbound Transport': inbound_transport_rate,
                    'Inbound Handling': total_pallets * 2 * self.parameters["inbound_handling_rate"],
                    'Storage': total_pallets * 2 * self.parameters["storage_rate"],
                    'Outbound Handling': total_pallets * 2 * self.parameters["outbound_handling_rate"],
                    'Outbound Transport': total_pallets * self.parameters["outbound_transport_rate"],
                    'Wooden Pallet': total_pallets * 2 * self.parameters["cost_per_wooden_pallet"],
                    'Other': total_pallets * self.parameters["other"],
                }
        else:
            components = {
                'Inbound Transport': inbound_transport_rate,
                'Inbound Handling': total_pallets * self.parameters["inbound_handling_rate"],
                'Storage': total_pallets * self.parameters["storage_rate"],
                'Outbound Handling': total_pallets * self.parameters["outbound_handling_rate"],
                'Outbound Transport': total_pallets * self.parameters["outbound_transport_rate"],
                'Wooden Pallet': total_pallets * self.parameters["cost_per_wooden_pallet"],
                'Other': total_pallets * self.parameters["other"],
            }

        total_cost = sum(components.values())

        if return_breakdown:
            # components['Total'] = total_cost
            return components
        return total_cost



    def format_label(self,value, delta):
        val_str = f"£{value/1000:.1f}K" if value >= 1000 else f"£{value:.0f}"
        delta_str = f" ({delta:+.1f}%)" if delta != 0 else ""
        return val_str + delta_str
    
    def plot_cost_comparison_plotly(self,cost_df):
        existing_cost = cost_df['Existing Cost'].iloc[0]
        cost_cols = [col for col in cost_df.columns if col.startswith("Cost")]

        optimized_data = []
        existing_data = []

        for col in cost_cols:
            value = cost_df[col].iloc[0]
            delta = ((value - existing_cost) / existing_cost) * 100 if existing_cost else 0

            if "Optimized Layer" in col:
                label = col.replace("Cost at ", "").replace(" with Optimized Layer", "")
                optimized_data.append({"label": label, "value": value, "delta": delta})
            elif "Existing Layer" in col:
                label = col.replace("Cost at ", "").replace(" with Existing Layer", "")
                existing_data.append({"label": label, "value": value, "delta": delta})

   

        # Plot for Optimized Layer
        fig1 = go.Figure()

        fig1.add_trace(go.Bar(
            x=["Existing Cost"],
            y=[existing_cost],
            name="Existing Cost",
            marker_color='indianred',
            text=[self.format_label(existing_cost, 0)],
            textposition="outside"
        ))

        for item in optimized_data:
            fig1.add_trace(go.Bar(
                x=[item['label']],
                y=[item['value']],
                name=f"Optimized Layer: {item['label']}",
                marker_color='steelblue',
                text=[self.format_label(item['value'], item['delta'])],
                textposition="outside"
            ))

        fig1.update_layout(
            title="Cost Comparison: Optimized Layers - " + self.mat + "( "+self.mat_name+" )",
            yaxis_title="Total Cost (£)",
            xaxis_title="Configuration",
            barmode='group',
            height=600,
            width=1200,
            showlegend=False
        )

        # Plot for Existing Layer
        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=["Existing Cost"],
            y=[existing_cost],
            name="Existing Cost",
            marker_color='indianred',
            text=[self.format_label(existing_cost, 0)],
            textposition="outside"
        ))

        for item in existing_data:
            fig2.add_trace(go.Bar(
                x=[item['label']],
                y=[item['value']],
                name=f"Existing Layer: {item['label']}",
                marker_color='orange',
                text=[self.format_label(item['value'], item['delta'])],
                textposition="outside"
            ))

        fig2.update_layout(
            title="Cost Comparison: Existing Layers - " + self.mat + "("+self.mat_name+")",
            yaxis_title="Total Cost (£)",
            xaxis_title="Configuration",
            barmode='group',
            height=600,
            width=1200,
            showlegend=False
        )

        # Display both charts
        # fig1.show()
        # fig2.show()

        return fig1, fig2

    def plot_clean_stacked_bar(self, df: pd.DataFrame, title: str = "Stacked Cost Breakdown (£)"):
        import re

        # Reset index to make it a column for plotting
        df_plot = df.reset_index().rename(columns={"index": "Cost Scenario"})
        
        # Melt the DataFrame to long format
        df_melted = df_plot.melt(id_vars="Cost Scenario", var_name="Cost Component", value_name="Cost")
        
        # Format cost values in thousands and add £K
        df_melted["Cost_k"] = (df_melted["Cost"] / 1000).round(0)
        df_melted["Cost Display"] = df_melted["Cost_k"].apply(lambda x: f"£{int(x)}K")

        # Smart shortening of scenario labels
        def smart_trim(s):
            # Remove 'Cost at' and split by 'with'
            s = s.replace("Cost at ", "")
            s = s.replace("Layer", "LYR")
            s = s.replace("Optimized", "Optimal")
            parts = s.split(" with ")
            # Abbreviate each part
            abbreviated = [re.sub(r'[aeiou]', '', p.split()[0])[:6] for p in parts if p]
            return " / ".join(parts)

        df_melted["Cost Scenario Short"] = df_melted["Cost Scenario"]
        df_melted["Cost Scenario Trimmed"] = df_melted["Cost Scenario"].apply(smart_trim)

        # Compute total cost per scenario for annotations
        total_costs = df_melted.groupby("Cost Scenario Trimmed")["Cost"].sum().reset_index()
        total_costs["Total Display"] = (total_costs["Cost"] / 1000).round(0).astype(int).astype(str).radd("£").add("K")

        # Plot
        fig = px.bar(
            df_melted,
            x="Cost Scenario Trimmed",
            y="Cost",
            color="Cost Component",
            text="Cost Display",
            title=title + f" - {self.mat}({self.mat_name})",
            hover_name="Cost Scenario Short",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        # Layout and styling
        fig.update_layout(
            barmode='stack',
            xaxis_tickangle=-30,
            height=800,
            width=1200,
            margin=dict(l=40, r=40, t=60, b=120),
            font=dict(size=12),
            legend_title_text='Component',
            plot_bgcolor='white',
            yaxis_title="Cost (£)",
        )

        # Show £K labels inside bars
        fig.update_traces(
            textposition='inside',
            insidetextanchor='middle'
        )

        # Add total annotations on top of bars
        for i, row in total_costs.iterrows():
            fig.add_annotation(
                x=row["Cost Scenario Trimmed"],
                y=row["Cost"],
                text=row["Total Display"],
                showarrow=False,
                font=dict(size=12, color="black", family="Arial"),
                yshift=10
            )

        return fig

    def format_row(self,row):
        metric = row.name  # row index is the metric name
        print(metric)
        
        # Apply formatting to all values except 'Metric'
        if metric == 'Total Cost (£)':
            return row.apply(lambda val: f"£{val / 1000:.0f}K")
        elif metric in ['Units per Pallet', 'Total Pallets Required','Shipper Boxes per Pallet']:
            return row.apply(lambda val: f"{int(val):,}")
        else:
            return row.astype(str)  
    def table_view(self,df, title="Metric Comparison"):

        df = df.apply(self.format_row, axis=1)
        # df['Shipper Boxes per Pallet'] = df['Shipper Boxes per Pallet'].astype(int)
        headers = [f"<b>{'Metric'}</b>"] + [f"<b>{col}</b>" for col in df.columns]
        values = [[str(i) for i in df.index]] + [df[col].tolist() for col in df.columns]
        
        # Alternate row colors
        num_rows = len(df)
        fill_colors = ['rgba(245, 245, 245, 1)' if i % 2 == 0 else 'white' for i in range(num_rows)]
        fill_color_matrix = [fill_colors] * len(values)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='rgb(230, 230, 230)',
                line_color='darkslategray',
                align='center',
                font=dict(color='black', size=13)
            ),
            cells=dict(
                values=values,
                fill_color=fill_color_matrix,
                line_color='lightgrey',
                align='center',
                font=dict(color='black', size=12),
                height=30
            )
        )])

        fig.update_layout(
            margin=dict(l=10, r=10, t=50 if title else 10, b=10),
            title=title+f" - {self.mat}({self.mat_name})",
            title_x=0.5,
            title_font=dict(size=18),
            height = 600,
            width = 1200
        )

        return fig


    def generate_results(self,llm,df,question):
        

        print("Parameters for Agent: ")
        for k,v in self.parameters.items():
            print(f"{k}:",v)
            
        # self.df = self.preprocessing_steps(df)
        self.df = df
        self.df['SKU'] = self.df['SKU'].astype(str)

        # print(self.df.columns)
        self.mat = self.parameters["mat"]
        self.pallet_type = self.parameters["pallet_type"]
        self._mat_df = self.df[self.df['SKU'] == self.mat]
        # print(self.mat)
        # print(self._mat_df.shape)
        self.mat_name = self._mat_df['Description'].values[0]

        
        result_df,breakdown_df,table_df, table_dict = self.calculate_pallet_config()
        fig1,fig2 = self.plot_cost_comparison_plotly(result_df)
        fig3 = self.plot_clean_stacked_bar(breakdown_df)
        fig4 = self.table_view(table_df.round(2), title="Metric Comparison")
        # print(result_df.columns)

        result = llm.invoke(
            f"""
            Given the following:
            - User question: {question}
            - Summary of pallet utilization scenarios: {table_dict}
            - Cost breakdown across all scenarios only required when a summaring a particular component cost such as Outbound, Inbound, Storage: {breakdown_df}

            Generate a **single professional and concise markdown-formatted summary** that:
            - Directly addresses the user’s question using relevant insights from all three inputs
            - Highlights the optimal scenario
            - Provides a clear comparison between scenarios
            - Lists all key extracted parameters in one sentence
            - Avoids unnecessary details, salutations or repetition
            - All the cost should be in £K format
            """
        )

        return {"final_response":result,"fig":[fig1,fig2,fig3,fig4]}