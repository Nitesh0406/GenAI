
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.load_templates import load_template


def get_final_prompt(question, is_sku_analysis):
    data_description = load_template("data_description.txt")
    KPI_description = load_template("KPI_description.txt")
    prompt = load_template("bi_agent_prompt.txt")
    KPI_correlation = load_template("kpi_correlation_matrix.csv")
    data_description_sku_master = load_template("sku_master_data_description.txt")
    sku_analysis_prompt = load_template("sku_analysis_prompt.txt")
    if is_sku_analysis == 'False':
        final_prompt = prompt.format(data_description=data_description, KPI_description=KPI_description, question=question,KPI_correlation=KPI_correlation)
    else:
        final_prompt = sku_analysis_prompt.format(data_description=data_description_sku_master,question=question)
    return final_prompt

def extract_code_segments(response_text):
    """Extract code segments from the API response using regex."""
    segments = {}
    
    # Extract approach section
    approach_match = re.search(r'<approach>(.*?)</approach>', response_text, re.DOTALL)
    if approach_match:
        segments['approach'] = approach_match.group(1).strip()
    
    # Extract content between <code> tags
    code_match = re.search(r'<code>(.*?)</code>', response_text, re.DOTALL)
    if code_match:
        segments['code'] = code_match.group(1).strip()
    
    # Extract content between <chart> tags
    chart_match = re.search(r'<chart>(.*?)</chart>', response_text, re.DOTALL)
    if chart_match:
        segments['chart'] = chart_match.group(1).strip()
    
    # Extract content between <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if answer_match:
        segments['answer'] = answer_match.group(1).strip()
    
    return segments

def extract_insight_llm_response(llm,question,is_sku_analysis):
    final_prompt = get_final_prompt(question, is_sku_analysis)
    response = llm.invoke(final_prompt)
    return response

def execute_python_code(namespace,segments,results):
    python_code_executed = False
    try:
        if 'code' in segments:
            # Properly dedent the code before execution
            code_lines = segments['code'].strip().split('\n')
            # Find minimum indentation
            min_indent = float('inf')
            for line in code_lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            # Remove consistent indentation
            dedented_code = '\n'.join(line[min_indent:] if line.strip() else '' for line in code_lines)
            print(f"Python Code : {dedented_code}")
            exec(dedented_code, namespace)
            result_df = namespace.get('result_df')
            print("executed python code")
            python_code_executed = True
        else:
            result_df = pd.DataFrame()
        return result_df,namespace,python_code_executed

    except Exception as e:
        result_df = pd.DataFrame()
        print("error in executing python code",e)
        return result_df,namespace, python_code_executed


def execute_chart_codes(namespace,segments,results):
    chart_code_executed = False
    try:
        if 'chart' in segments:
                # Properly dedent the chart code
                chart_lines = segments['chart'].strip().split('\n')
                # Find minimum indentation
                min_indent = float('inf')
                for line in chart_lines:
                    if line.strip():  # Skip empty lines
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)
                # Remove consistent indentation
                dedented_chart = '\n'.join(line[min_indent:] if line.strip() else '' for line in chart_lines)
                dedented_chart = dedented_chart.replace("f'£{value:,.2f}'", '"£"+format_large_value(value)')
                dedented_chart = dedented_chart.replace("f'£{x:,.2f}'", '"£"+format_large_value(x)')
                dedented_chart = dedented_chart.replace('f"{x:,.0f}"', 'format_large_value(x)')
                if ("type='category'" not in dedented_chart) and ("type = 'category'" not in dedented_chart):
                    dedented_chart = dedented_chart.replace(
                        "xaxis=dict(",
                        "xaxis=dict(\n        type='category',"
                    )
                print(f"Chart Code : {dedented_chart}")
                exec(dedented_chart, namespace)
                print("executed chart code")
                chart_code_executed = True
                fig = namespace.get('fig')
                if fig:
                    fig.update_layout(width=1000, height=600)
                    results['figure'] = fig
        return results, chart_code_executed
    except Exception as e:
        results['figure'] = None
        print("error in executing chart code",e)
        return results, chart_code_executed

def format_large_value(num):
    """
    Converts large numbers in short human-readable text
    """
    if isinstance(num, int) or isinstance(num, float):
        if num > 1000000:
            return str(round(num/1000000, 1)) + 'M'
        elif num > 1000:
            return str(round(num/1000, 1)) + 'K'
        else:
            return str(round(num,2))
    else:
        return str(round(num,2))

def prepare_final_result(llm, question, results, result_df, is_sku_analysis, chart_code_executed):
    if not result_df.empty:
        python_code = results['code']
        data_description = load_template("data_description.txt")
        KPI_description = load_template("KPI_description.txt")
        KPI_correlation = load_template("kpi_correlation_matrix.csv")
        prompt = load_template("bi_agent_answer_prompt.txt")
        data_description_sku_master = load_template("sku_master_data_description.txt")
        sku_analysis_ans_prompt = load_template('sku_analysis_answer_prompt.txt')
        if is_sku_analysis == 'False':
            ans_prompt = prompt.format(question=question, python_code=python_code, result_df=result_df, data_description=data_description, KPI_description=KPI_description, KPI_correlation=KPI_correlation)
            response = llm.invoke(ans_prompt)
            response_text = response.content
        else:
            ans_prompt = sku_analysis_ans_prompt.format(question=question, python_code=python_code, result_df=result_df, data_description=data_description_sku_master)
            response = llm.invoke(ans_prompt)
            response_text = response.content
        # print(f"Answer Response Text : {response_text}")
        answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if answer_match:
            results['answer'] = answer_match.group(1).strip()
    else:
        if chart_code_executed:
            results['answer'] = "Please refer above chart for your answer."
        else:
            results['answer'] = "Unable to extract answer for given query due to limitations in bot capabilities."
    return results

def prepare_sku_master(sku_master_df):
    ## renaming columns
    column_name_mapping = {
        "SKU": "SKU_CODE",
        "Description": "SKU_DESCRIPTION",
        "Supplier": "SUPPLIER",
        "Location": "LOCATION",
        "Pack Config": "PACK_CONFIG",
        "Volume": "SHIPMENT_VOLUME",
        "ST/PLT": "ITEMS_PER_PALLET",
        "ST/SHP": "ITEMS_PER_SHIPPER",
        "ST/LYR": "ITEMS_PER_LAYER",
        "LYR/PLT": "LAYERS_PER_PALLET",
        "SHP/PLT": "SHIPPERS_PER_PALLET",
        "SHP/LYR": "SHIPPERS_PER_LAYER",
        "EACH_HEIGHT (mm)": "ITEM_HEIGHT (mm)",
        "EACH_WIDTH (mm)": "ITEM_WIDTH (mm)",
        "EACH_LENGTH (mm)": "ITEM_LENGTH (mm)",
        "SHIPPER_HEIGHT (mm)": "SHIPPER_HEIGHT (mm)",
        "SHIPPER_LENGTH (mm)": "SHIPPER_LENGTH (mm)",
        "SHIPPER_WIDTH (mm)": "SHIPPER_WIDTH (mm)",
        "PLT_HEIGHT (mm)": "PALLET_HEIGHT (mm)",
        "PLT_LENGTH (mm)": "PALLET_LENGTH (mm)",
        "PLT_WIDTH (mm)": "PALLET_WIDTH (mm)",
        "EACH_WT": "ITEM_WEIGHT(KG)",
        "SHIPPER_WT": "SHIPPER_WEIGHT(KG)",
        "PLT_WT": "PALLET_WEIGHT(KG)",
        "Pallet Utilization % (IND)": "PALLET_UTILIZATION%_(IND)",
        "Pallet Utilization % (EURO)": "PALLET_UTILIZATION%_(EURO)",
        "Layer Utilization % (IND)": "LAYER_UTILIZATION%_(IND)",
        "Layer Utilization % (EURO)": "LAYER_UTILIZATION%_(EURO)",
        "Shipper Utilization %": "SHIPPER_UTILIZATION%",
    }
    sku_master_df = sku_master_df.rename(columns=column_name_mapping)
    return sku_master_df

def execute_codes(llm, question, insight_df,sku_master_df, response_text, is_sku_analysis):
    """Execute the extracted code segments on the provided dataframe and store formatted answer."""
    results = {
        'approach': None,
        'answer': None,
        'figure': None,
        'code': None,
        'chart_code': None,
        'result_df': None
    }
    try:
        # Extract code segments
        segments = extract_code_segments(response_text)
        if not segments:
            print("No code segments found in the response")
            return results
        # Store the approach and raw code
        if 'approach' in segments:
            results['approach'] = segments['approach']
        if 'code' in segments:
            results['code'] = segments['code']
        if 'chart' in segments:
            results['chart_code'] = segments['chart']
        if is_sku_analysis == 'False':
            # Create a single namespace for all executions
            namespace = {'df': insight_df, 'pd': pd, 'plt': plt, 'sns': sns,'np':np, 'format_large_value':format_large_value}
            result_df,namespace, python_code_executed = execute_python_code(namespace,segments,results)
            results, chart_code_executed = execute_chart_codes(namespace,segments,results)
            results = prepare_final_result(llm, question, results, result_df, is_sku_analysis, chart_code_executed)
        else:
            sku_master_df_final = prepare_sku_master(sku_master_df)
            namespace = {'df': sku_master_df_final, 'pd': pd, 'plt': plt, 'sns': sns,'np':np, 'format_large_value':format_large_value}
            result_df,namespace, python_code_executed = execute_python_code(namespace,segments,results)
            results,chart_code_executed = execute_chart_codes(namespace,segments,results)
            results = prepare_final_result(llm, question, results, result_df, is_sku_analysis, chart_code_executed)
        results['result_df'] = result_df
        return results
    except Exception as e:
        print("error",e)
        return results
