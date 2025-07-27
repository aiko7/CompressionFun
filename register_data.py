from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from mlclient import ml_client
data = Data(
    path="./tiny-imagenet-200",    # local , data already registered
    name="tiny-imagenet-200",
    version="6",
    type=AssetTypes.URI_FOLDER
)

ml_client.data.create_or_update(data)
