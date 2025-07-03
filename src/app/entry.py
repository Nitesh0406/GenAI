"""
ui2.py

Streamlit-based UI for the multi-agent generative AI system.
"""
import os
import time
import tiktoken
import openai
import base64
from pymongo import MongoClient
from langchain_core.messages import HumanMessage
from src.agents.supervisor import get_supervisor_chain_for_follow_ques
from src.utils.get_examples import generate_description_and_followup_question
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from src.orchestrater.MultiAgentGraph import create_agent_graph
import streamlit as st
from langgraph.checkpoint.mongodb import MongoDBSaver


# ---------------------- Helper Functions ----------------------

def display_saved_plot(plot_path: str):
    """
    Loads and prints the path of a saved plot.

    Args:
        plot_path (str): Path to the saved plot image.
    """
    if os.path.exists(plot_path):
        print(f"✅ Plot saved at: {plot_path}")
    else:
        print(f"❌ Plot not found at {plot_path}")

def load_data_file():
    """Load data from a CSV file."""
    file_path = os.path.join("src", "data", "Outbound_Data.csv")
    if not os.path.exists(file_path):
        return None
    return file_path

def summarize_messages(messages):
    """Summarizes a long conversation to maintain token limit."""
    prompt = "Summarize the following conversation in a concise way:\n\n"
    for msg in messages:
        prompt += f"{msg.content}\n"

    summary = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                  {"role": "user", "content": prompt}]
    )
    return summary.choices[0].message.content

def num_tokens_from_messages(messages, model="gpt-4o"):

    """Return the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)

    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4

        num_tokens += len(encoding.encode(message.content))
        num_tokens -= 1  # Role is always required and always 1 token
        num_tokens += 2  # Every reply is primed with <im_start>assistant

    return num_tokens

def setup_data_source():
    """Automatically selects a single default data source without user input."""
    api_key = os.getenv("OPENAI_API_KEY", "")

    # Automatically pick the default data file
    data_file = os.path.join("src", "data", "Outbound_Data.csv")
    print(f"\n📂 Using default data source: Outbound_Data.csv")

    return api_key, data_file

def encode_image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def delete_conversation_for_thread_and_user(thread_id,user_id,MONGODB_URI,DB_NAME):
    """
        Deletes all conversation documents from the 'checkpoints_aio' collection in the specified MongoDB database
        that match a given `thread_id` and `user_id`.

        Parameters:
            thread_id (str): The identifier of the conversation thread to delete.
            user_id (str): The user ID associated with the conversation.
            MONGODB_URI (str): The connection URI string for MongoDB.
            DB_NAME (str): The name of the MongoDB database.

        Behavior:
            - Connects to the MongoDB instance using the provided URI and database name.
            - Filters documents in the 'checkpoints_aio' collection based on the given thread ID and user ID (as a byte-encoded string).
            - Prints the number of documents found matching the filter.
            - Deletes all matching documents and prints how many were deleted.

        Notes:
            The user_id is matched using a byte string format which may be specific to your storage schema.
    """

    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]  # Replace with your DB name


    # Access a specific collection (change based on your structure)
    collection = db["checkpoints_aio"] 

    query = {
    "thread_id": thread_id,
    "metadata.user_id": bytes(f'"{user_id}"', 'utf-8') # Note the byte string format
    }

    filtered_docs = list(collection.find(query))
    print(f"Found {len(filtered_docs)} matching documents.")

    # Perform the deletion based on the query
    result = collection.delete_many(query)
    
    # Print how many documents were deleted
    print(f"Deleted {result.deleted_count} documents.")

# async def run_app(llm, question: str, thread_id: str,user_id:str,shipment_df,rate_card,insights_df,SKU_master,azure_client):
#     """Main UI function to handle user interactions and execute the multi-agent graph."""
#     print("UK Distribution CTS Insights & Optimisation Agent")
#     print("Initializing system...")
#     MONGODB_URI = "mongodb+srv://copilotgenai:CFMEEGr0T6O63wuk@clusterai.nd3uoy.mongodb.net/?retryWrites=true&w=majority&appName=Clusterai"
#     DB_NAME = "checkpointing_db"
#     async with AsyncMongoDBSaver.from_conn_string(MONGODB_URI, db_name=DB_NAME) as checkpointer:
#         multi_agent_graph = await create_agent_graph(llm=llm,shipment_df=shipment_df,rate_card=rate_card,insights_df=insights_df,SKU_master=SKU_master,checkpointer=checkpointer,azure_client=azure_client)
#         config = {"configurable": {"thread_id": thread_id,"user_id":user_id}}
#         print("\nSystem ready. You can start asking questions.")
#         start_time = time.time()
#         state = {
#             "messages": [HumanMessage(content=question)],
#             "next": "supervisor",
#             "visual_outputs": [],
#             "visual_image":[],
#             "current_agent": None,
#             "metadata": {},
#             "parameters": None  # Initialize parameters as None
#         }
#         answers = []
#         if question.lower() in ["start a new session","hi","hello","good afternoon", "good morning","good evening","hey"]:
#             delete_conversation_for_thread_and_user(thread_id,user_id,MONGODB_URI,DB_NAME)
#             # answers = [f"Deleted current `thread_id`: {thread_id} and `user_id`: {user_id} from database..!"]
#             answer_content = "System has been starting a new conversation.\n I am your dedicated logistics agent, equipped to support you with data analysis and optimization solutions.\n To learn more about my capabilities, please explore the questions below."
#             answers = [{'agent': 'Supervisor', 'text': answer_content}]
#             (f"\nAnalysis completed in {time.time() - start_time:.1f} seconds")
#             return {
#                     'messages': answers,
#                     'charts': [],
#                     'status': 'success',
#                     'follow_up':['Please tell about your capabilities and datasets',
#                                  'Brief me about key agents and optimisation strategies available',
#                                  'Provide me some examples of Insight Questions',
#                                  'Provide me some examples of Order Frequency Optimization Questions',
#                                  'Provide me some examples of Drop Location Centralization Questions',
#                                  'Provide me some examples of Pallet Utilization Questions']
#
#                     }
#
#         elif question in ['Please tell about your capabilities and datasets',
#                                  'Brief me about key agents and optimisation strategies available',
#                                  'Provide me some examples of Insight Questions',
#                                  'Provide me some examples of Order Frequency Optimization Questions',
#                                  'Provide me some examples of Drop Location Centralization Questions',
#                                  'Provide me some examples of Pallet Utilization Questions']:
#
#             answer_content,followup_questions = generate_description_and_followup_question(question)
#
#             # answer_content = "Development in progress, please try again later."
#             answers = [{'agent': 'Supervisor', 'text': str(answer_content)}]
#             return {
#                     'messages': answers,
#                     'charts': [],
#                     'status': 'success',
#                     'follow_up':followup_questions
#                     }
#
#         agents_calling_list = []
#         async for current_state in multi_agent_graph.astream(state, config):
#             st.write("Inside entry stream.py")
#             # print("Current State ------------------ \n",current_state)
#             # state['messages'] = add_messages(state['messages'], current_state['messages'])
#             if agents_calling_list == ['supervisor','InsightsAgent','supervisor','InsightsAgent'] or len(agents_calling_list)>6:
#                 break
#             if isinstance(current_state, dict) and 'next' in current_state:
#                 state['next'] = current_state['next']
#             if isinstance(current_state, dict) and 'parameters' in current_state:
#                 state['parameters'] = current_state['parameters']
#
#             section = list(current_state.values())[0]
#             message = section['messages'][0]
#             # Extract content and name
#             content = message.content
#             name = message.name
#             if name == 'supervisor':
#                 name = 'Supervisor'
#             answers.append({"agent": name, "text":content})
#             agents_calling_list.append(name)
#             print("ending current state loop")
#     print(f"\nAnalysis completed in {time.time() - start_time:.1f} seconds")
#     print("Printing state at the end",state)
#     charts = []
#     images = []
#     if state['visual_outputs'] is not None:
#         for path in state['visual_outputs']:
#             charts.append({'content':path})
#     if state['visual_image'] is not None:
#         for img in state['visual_image']:
#             images.append(img)
#
#     # print("-"*len("Please wait, Generating follow-up questions...!"))
#     # print("Please wait, Generating follow-up questions...!")
#
#     follow_up_prompt_file_path = os.path.join("prompt_templates","follow_up_questions.txt")
#     # Open the file in read mode
#     with open(follow_up_prompt_file_path, "r") as file:
#         follow_up_prompt_file_path = file.read()
#
#     follow_up_ques_chain = get_supervisor_chain_for_follow_ques()
#     res = follow_up_ques_chain.invoke({"input":"","prompt":follow_up_prompt_file_path,"question":question,"answer":answers})
#
#     # print("-"*len("Thought process while generating follow-up: "))
#     # print("Thought process while generating follow-up: ")
#     # print("-"*len("Thought process while generating follow-up: "))
#     # print(res['thought_process'])
#     # print("-"*len("Next logical steps to follow: "))
#     # print("Next logical steps to follow: ")
#     # print(res.get("next_logical_flow",""))
#     # Adding the message to the last message provided by the supervisor
#     answers.append({"agent": "Suggested Next Steps", "text":res.get("next_logical_flow","")})
#     # print("-"*70)
#     # print("Follow-up questions generated: ", res["follow_up_questions"])
#
#     return {
#         'messages': answers,
#         'charts': charts,
#         'images': images,
#         'follow_up': res["follow_up_questions"],
#         'status': 'success'
#     }


def run_app(llm, question: str, thread_id: str, user_id: str, shipment_df, rate_card, insights_df, SKU_master,
            azure_client):
    """Main UI function to handle user interactions and execute the multi-agent graph."""
    print("UK Distribution CTS Insights & Optimisation Agent")
    print("Initializing system...")

    MONGODB_URI = "mongodb+srv://copilotgenai:CFMEEGr0T6O63wuk@clusterai.nd3uoy.mongodb.net/?retryWrites=true&w=majority&appName=Clusterai"
    DB_NAME = "checkpointing_db"

    with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME) as checkpointer:

        workflow = create_agent_graph(
            llm=llm,
            shipment_df=shipment_df,
            rate_card=rate_card,
            insights_df=insights_df,
            SKU_master=SKU_master,
            # checkpointer=checkpointer,
            azure_client=azure_client
        )
        multi_agent_graph = workflow.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        print("\nSystem ready. You can start asking questions.")
        start_time = time.time()

        state = {
            "messages": [HumanMessage(content=question)],
            "next": "supervisor",
            "visual_outputs": [],
            "visual_image": [],
            "current_agent": None,
            "metadata": {},
            "parameters": None
        }

        answers = []

        if question.lower() in ["start a new session", "hi", "hello", "good afternoon", "good morning", "good evening",
                                "hey"]:
            delete_conversation_for_thread_and_user(thread_id, user_id, MONGODB_URI, DB_NAME)
            answer_content = (
                "System has been starting a new conversation.\n"
                "I am your dedicated logistics agent, equipped to support you with data analysis and optimization solutions.\n"
                "To learn more about my capabilities, please explore the questions below."
            )
            answers = [{'agent': 'Supervisor', 'text': answer_content}]
            return {
                'messages': answers,
                'charts': [],
                'status': 'success',
                'follow_up': [
                    'Please tell about your capabilities and datasets',
                    'Brief me about key agents and optimisation strategies available',
                    'Provide me some examples of Insight Questions',
                    'Provide me some examples of Order Frequency Optimization Questions',
                    'Provide me some examples of Drop Location Centralization Questions',
                    'Provide me some examples of Pallet Utilization Questions'
                ]
            }

        elif question in [
            'Please tell about your capabilities and datasets',
            'Brief me about key agents and optimisation strategies available',
            'Provide me some examples of Insight Questions',
            'Provide me some examples of Order Frequency Optimization Questions',
            'Provide me some examples of Drop Location Centralization Questions',
            'Provide me some examples of Pallet Utilization Questions'
        ]:
            answer_content, followup_questions = generate_description_and_followup_question(question)
            answers = [{'agent': 'Supervisor', 'text': str(answer_content)}]
            return {
                'messages': answers,
                'charts': [],
                'status': 'success',
                'follow_up': followup_questions
            }

        # Replacing async for loop with sync equivalent
        agents_calling_list = []
        for current_state in multi_agent_graph.stream(state, config):  # NOTE: use `.stream()` instead of `.astream()`
            if agents_calling_list == ['supervisor', 'InsightsAgent', 'supervisor', 'InsightsAgent'] or len(
                    agents_calling_list) > 6:
                break

            if isinstance(current_state, dict) and 'next' in current_state:
                state['next'] = current_state['next']
            if isinstance(current_state, dict) and 'parameters' in current_state:
                state['parameters'] = current_state['parameters']

            section = list(current_state.values())[0]
            message = section['messages'][0]
            content = message.content
            name = message.name
            if name == 'supervisor':
                name = 'Supervisor'
            answers.append({"agent": name, "text": content})
            agents_calling_list.append(name)
            print("ending current state loop")

        print(f"\nAnalysis completed in {time.time() - start_time:.1f} seconds")
        print("Printing state at the end", state)

        charts = []
        images = []
        if state['visual_outputs'] is not None:
            for path in state['visual_outputs']:
                charts.append({'content': path})
        if state['visual_image'] is not None:
            for img in state['visual_image']:
                images.append(img)

        follow_up_prompt_file_path = os.path.join("prompt_templates", "follow_up_questions.txt")
        with open(follow_up_prompt_file_path, "r") as file:
            follow_up_prompt_text = file.read()

        follow_up_ques_chain = get_supervisor_chain_for_follow_ques()
        res = follow_up_ques_chain.invoke({
            "input": "",
            "prompt": follow_up_prompt_text,
            "question": question,
            "answer": answers
        })

        answers.append({
            "agent": "Suggested Next Steps",
            "text": res.get("next_logical_flow", "")
        })

    return {
        'messages': answers,
        'charts': charts,
        'images': images,
        'follow_up': res["follow_up_questions"],
        'status': 'success'
    }
