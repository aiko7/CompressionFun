from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Output
import os 

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
)


for class_id in range(200):
    job = command(
        name=f"autoenc-job-{class_id}",
        display_name=f"Autoencoder Class {class_id}",
        experiment_name="ae-compression-exp",
        description=f"Training autoencoder on class {class_id}",
        code=".",  # entire project folder packaged once
        command=(
            "poetry run python train.py "
            f"--class_id {class_id} "
            "--checkpoint_dir ${{outputs.checkpoints}}"
        ),
        environment="autoencoder-env@latest",
        compute="gpu-spot-cluster",
        distribution={"type": "pytorch", "process_count_per_instance": 1},
        outputs={
            "checkpoints": Output(
                type="uri_folder",
                mode="rw_mount",  # keep checkpoints across retries
                path=(
                    "azureml://datastores/workspaceblobstore/"
                    f"paths/ae_ckpts/class_{class_id}"
                ),
            )
        },
        limits={"retry_settings": {"max_retries": 3}},
    )

    out_job = ml_client.jobs.create_or_update(job)
    print(f"Submitted job {out_job.name} for class {class_id}")
