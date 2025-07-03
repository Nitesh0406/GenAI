from fastapi import FastAPI
from azure.storage.blob import BlobServiceClient, ContentSettings
from datetime import datetime,timedelta
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from io import BytesIO

class AzureBlobStorage:
    def __init__(self, connection_string: str, container_name: str, blob_url_base: str):
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.blob_url_base = blob_url_base

    def upload_image_to_blob(self,image_bytes_io: BytesIO, filename: str, content_type: str) -> str:
        blob_client = self.container_client.get_blob_client(filename)
        blob_client.upload_blob(
            image_bytes_io.getvalue(),
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type))
        
        sas_token = generate_blob_sas(
        account_name="strcopilotlogisticsgenai",
        container_name=self.container_name,
        blob_name=filename,
        account_key="DA4tYeyTFSTE8pIJEWBKt1ZPsqOY8xS1DXtpCGg2MhAhzUfuKDT6PYFN9E2Hg5BwlyPwBZIuNf+g+AStKYn7hA==",  # Get this from Azure Portal
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1))
        
        return f"https://strcopilotlogisticsgenai.blob.core.windows.net/{self.container_name}/{filename}?{sas_token}"
        # return f"{self.blob_url_base}/{filename}"