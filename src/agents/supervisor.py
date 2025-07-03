import json
import os
from dotenv import load_dotenv, find_dotenv
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.utils.openai_api import get_supervisor_llm
from src.utils.load_templates import load_template
from langchain_community.chat_models import ChatOpenAI

def get_supervisor_chain(agents_file_path: str = "./prompt_templates/agents_description.json"):
    # Load environment variables
    _ = load_dotenv(find_dotenv())

    # Load agent definitions from JSON
    with open(agents_file_path, 'r') as f:
        members = json.load(f)

    supervisor_prompt = load_template("supervisor_prompt.txt")
    sku_master_data_desc = load_template("sku_master_data_description.txt")
    sku_analysis_instruction = load_template("sku_analysis_instruction.txt")
    llm = get_supervisor_llm()

    # Define available options for the supervisor to choose from
    options = ["FINISH"] + [mem["agent_name"] for mem in members]

    # Generate structured agent information
    members_info = "\n".join([f"{member['agent_name']}: {member['description']}" for member in members])

    # Format the full prompt
    final_prompt = supervisor_prompt + "\nHere is the information about the different agents available:\n" + members_info
    final_prompt = final_prompt + "\nHere is data description for SKU Master data\n" + sku_master_data_desc
    final_prompt += """
                    Think step-by-step before choosing the next agent or deciding to answer directly. 
                    Examples of when reasoning_task is True/False:
                    - What is the trend of <KPI> in year 2024? [reasoning_task will be False since query is related facts]
                    - Why the trend is declining? [reasoning_task will be True since query is related to why did it happened.]
                    Examples of when to use SELF_RESPONSE:
                    - "Can you explain what the Insights Agent does?"
                    - "What kind of data does this system analyze?"
                    - "I'm not sure how to phrase my question about cost optimization"
                    - "What's the difference between Static and Dynamic Cost Optimization?"
                    - "Provide me customers selected from last question"
                    - "Provide me parameters for the last question"

                    Examples of when to route to specialized agents:
                    - "Analyze the shipment data and tell me which postcode has the highest delivery cost" (Insights Agent)
                    - "How can we optimize our delivery schedule to reduce costs?" (Cost Optimization Agents)
                    - "Run a drop point centralization for TESCO STORES LTD with 4 drop points." (Drop Point Centralization Optimization Agent)
                    - "Find the top 5 drop points for TESCO STORES LTD between 2025-01-01 and 2025-03-01, rank them based on Distance." (Drop Point Centralization Optimization Agen)

                    If needed, reflect on responses and adjust your approach and finally provide response.
                    """
    final_prompt = final_prompt + "\n\n" + sku_analysis_instruction
    # Create LangChain prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", final_prompt.strip()),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Define the routing function schema
    function_def = {
        "name": "route",
        "description": "Select the next role based on reasoning.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "thought_process": {
                    "title": "Thought Process and Response",
                    "type": "string",
                    "description": "Step-by-step reasoning behind the decision and reply to the question."
                },
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}],
                    "description": "The next agent to call or SELF_RESPONSE if answering directly."
                },
                "reasoning_task": {
                    "title": "Reasoning Task",
                    "type": "string",
                    "description": "Default value is False, True only when question is related to reasoning (why did it happened) and next agent is `Insights Agent`."
                },
                "sku_analysis": {
                    "title": "Analysis related to material or sku",
                    "type": "string",
                    "description": "Default value is False, True only when question can be answered using sku master data."
                },
                "direct_response": {
                    "title": "Direct Response",
                    "type": "string",
                    "description": "The direct response to provide to the user when SELF_RESPONSE is selected."
                },
                "enriched_question": {
                    "title": "Enriched Question",
                    "type": "string",
                    "description": """By considering all the previous messages or conversation and the next agent to be called, frame a single line question.
                    Keep track of these parameters while summarising:
                    start_date;
                    end_date;
                    group_method;
                    all_post_code; 
                    all_customers;
                    selected_postcodes; 
                    selected_customers;
                    scenario;
                    shipment_window_range;
                    total_shipment_capacity;
                    utilization_threshold;"""
                },
            },
            "required": ["thought_process", "next", "direct_response","enriched_question","reasoning_task","sku_analysis"],
        },
    }

    # Create and return the supervisor chain
    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    return supervisor_chain,members

def get_supervisor_chain_for_follow_ques():
    # Load environment variables
    _ = load_dotenv(find_dotenv())

    llm = get_supervisor_llm()

    llm = ChatOpenAI(
    model_name="gpt-4o",  # your desired model
    temperature=0.5,           # your new temperature
    openai_api_key=llm.openai_api_key.get_secret_value(),  # reuse original config
    model_kwargs=llm.model_kwargs       # reuse other kwargs if needed
    )


    function_def = {
        "name": "follow_up_route",
        "description": "Choose next agent or answer directly, with step-by-step reasoning and a fully parameterized follow-up question.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought_process": {
                    "type": "string",
                    "description": "1–5: your numbered reasoning summary. Wha is your underlying thought process while generating the response."
                },
                "next_logical_flow": {
                    "type": "string",
                    "description": """A plain-suggestion message that guides the user toward the next logical step or question, based on the agent's thought process.
                    This message should appear as an informational suggestion at the bottom of the agent's response, helping the user continue the conversation meaningfully. Also make sure that this suggestion should be slightly comprehensive and should help user to choose next question precisely.
                    Try to generate your response by beginning with following phrases (not limited to these, you can explopre more):
                    Examples: 'Based on the analysis, I recommend exploring the following KPIs next.' or 'Would you like to proceed with optimizing the cost for this customer?'"""
                },
                "direct_response": {
                    "type": "string",
                    "description": "Your direct answer if using SELF_RESPONSE (empty otherwise)."
                },
                "follow_up_questions": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "minItems": 5,
                    "maxItems": 5,
                    "description": "List of 5 follow-up questions the user might ask based on the current exchange."
                }
            },
            "required": [
                "thought_process", "direct_response", "follow_up_questions"
            ]
        }
    }
    
    follow_up_ques_prompt = ChatPromptTemplate.from_messages([
        ("system", """{prompt}
        Last question asked by the user: {question}
        Response received by the user: {answer}
        """),
        ("human", "{input}")
        ])
    # Create and return the supervisor chain
    follow_up_ques_chain = (
        follow_up_ques_prompt
        | llm.bind_functions(functions=[function_def], function_call="follow_up_route")
        | JsonOutputFunctionsParser()
    )
    return follow_up_ques_chain