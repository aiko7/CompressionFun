from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import os
from dotenv import load_dotenv

load_dotenv()

subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
resource_group_name=os.getenv('AZURE_RESOURCE_GROUP')
workspace_name=os.getenv('AZURE_WORKSPACE_NAME')

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name
)