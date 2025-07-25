from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from mlclient import ml_client
data = Data(
    path="azureml://datastores/workspaceblobstore/paths/tiny-imagenet-200/",
    name="tiny-imagenet-200",
    version="2",
    type=AssetTypes.URI_FOLDER
)

ml_client.data.create_or_update(data)
