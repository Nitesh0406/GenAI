"""
config.py

Centralized configuration utilities for the generative AI project.
"""

import os
import logging
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())


# === Azure Blob Configuration ===
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=strcopilotlogisticsgenai;AccountKey=DA4tYeyTFSTE8pIJEWBKt1ZPsqOY8xS1DXtpCGg2MhAhzUfuKDT6PYFN9E2Hg5BwlyPwBZIuNf+g+AStKYn7hA==;EndpointSuffix=core.windows.net"
AZURE_CONTAINER_NAME = "responseimages"
AZURE_BLOB_URL_BASE = f"https://strcopilotlogisticsgenai.blob.core.windows.net/{AZURE_CONTAINER_NAME}"

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SHIPMENT_DF_PATH = os.path.join("src", "data", "Complete Input.xlsx")
RATECARD_PATH = os.path.join("src", "data", "Cost per pallet.xlsx")
INSIGHTS_DATA_PATH = os.path.join("src", "data", "Insights_Data.csv")
SKU_MASTER_PATH = os.path.join("src", "data", "SKU_master.csv")
OPENAI_MODEL = "gpt-4o"  # Make sure this matches your API access
MAX_TOKENS = 10000
TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
FUZZY_THRESHOLD = float(os.environ.get("FUZZY_THRESHOLD", 80))
# -----------------------------------------------------------------------------
# 2. LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
def setup_logging():
    """
    Sets up logging configuration with file name, function name, and line number.
    Logs messages to the console at INFO level.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("Logging initialized.")
