"""
entrypoint.py - FastAPI app initialization
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import matplotlib
import platform
import pandas as pd
from api.routes import router
from config.config import SHIPMENT_DF_PATH,RATECARD_PATH,INSIGHTS_DATA_PATH,SKU_MASTER_PATH
from config.config import AZURE_CONNECTION_STRING,AZURE_CONTAINER_NAME,AZURE_BLOB_URL_BASE
from src.utils.blob_access import AzureBlobStorage
if platform.system() == 'Darwin':  # macOS
    matplotlib.use('Agg')

def create_application() -> FastAPI:
    app = FastAPI(
        title="Multi-Agent AI System API",
        description="API for the UK Distribution CTS Insights & Optimisation Agent"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.on_event("startup")
    async def load_data():
        app.state.azure_client = AzureBlobStorage(
            connection_string=AZURE_CONNECTION_STRING,
            container_name=AZURE_CONTAINER_NAME,
            blob_url_base=AZURE_BLOB_URL_BASE
        )
        app.state.shipment_df = pd.read_excel(SHIPMENT_DF_PATH, sheet_name="Sheet1")
        app.state.rate_card = pd.read_excel(RATECARD_PATH)
        app.state.insights_df = pd.read_csv(INSIGHTS_DATA_PATH)
        app.state.sku_master = pd.read_csv(SKU_MASTER_PATH)
        print("✅ Loaded shipment_df and rate_card into app.state")

    app.include_router(router)
    return app

app = create_application()
