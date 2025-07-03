"""
routes.py - API endpoints
"""
from fastapi import APIRouter, HTTPException
from api.models import QueryRequest, QueryResponse
from src.utils.openai_api import get_supervisor_llm
from src.app.entry import run_app
import os
import dotenv
from fastapi import Request

# Load environment variables from .env file
dotenv.load_dotenv()

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest, data_request: Request):
    """
    Process user query through the multi-agent system.
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    print("query hit")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment")

    try:
        print("api key found--------------------------------------------\n")
        llm = get_supervisor_llm(api_key)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid API Key: {e}")

    try:
        shipment_df = data_request.app.state.shipment_df
        rate_card = data_request.app.state.rate_card
        insights_df = data_request.app.state.insights_df
        SKU_master = data_request.app.state.sku_master
        azure_client = data_request.app.state.azure_client

        result = await run_app(llm=llm, question=request.question, thread_id=request.thread_id,user_id=request.user_id,
                               shipment_df=shipment_df,rate_card=rate_card,insights_df=insights_df,SKU_master=SKU_master,azure_client=azure_client)
        print("data read completed...")
        follow_up_list = []
        for i in range(len(result['follow_up'])):
            follow_up_list.append({"id": str(i + 1), "DisplayName": result['follow_up'][i]})
        follow_up_list.append({"id": str(len(result['follow_up']) + 1), "DisplayName": "Start a New Session"})
        return QueryResponse(
            status="success",
            messages=result["messages"],
            visual_outputs=result['charts'],
            follow_up = follow_up_list,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
