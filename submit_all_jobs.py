import time
import argparse
from azure.ai.ml import command, Output, Input
from azure.ai.ml.constants import AssetTypes
from mlclient import ml_client

parser = argparse.ArgumentParser(description="Job sending configuration")
parser.add_argument("--classes", type=int, default=200, help="Number of classes to use fromm imagenet200, default=200")

args = parser.parse_args()
try:
    env = ml_client.environments.get("poetry-env",version="21")
    print(f"Environment found: {env.name}")
except Exception as e:
    print(f"Environment not found: {e}")
    print("Please create the environment first using: az ml environment create --file .azureml/environment.yaml")
    exit(1)

try:
    compute = ml_client.compute.get("autoencoder-gpu-cluster")
    print(f"Compute found: {compute.name}")
except Exception as e:
    print(f"Compute not found: {e}")
    print("Please create the compute first using: az ml compute create --file .azureml/compute.yaml")
    exit(1)

successful_submissions = 0
failed_submissions = 0

for class_id in range(args.classes):
    job = command(
        display_name=f"Autoencoder Class {class_id}",
        experiment_name="ae-compression-exp",
        description=f"Training autoencoder on class {class_id}",
        code=".",
        command=(
            "python train.py "
            f"--class_id {class_id} "
            "--checkpoint_dir ${{outputs.checkpoints}} "
            "--data_dir ${{inputs.data}}"
        ),
        environment="azureml:poetry-env:21",
        compute="autoencoder-gpu-cluster",
        inputs={
            "data": Input(
                type=AssetTypes.URI_FOLDER,
                path="azureml:tiny-imagenet-200:6"
            )
        },
        outputs={
            "checkpoints": Output(
                type=AssetTypes.URI_FOLDER,
                mode="rw_mount",
                path=f"azureml://datastores/workspaceblobstore/paths/ae_ckpts/class_{class_id}"
            )
        }
    )

    try:
        out_job = ml_client.jobs.create_or_update(job)
        print(f"✓ Submitted job {out_job.name} for class {class_id}")
        successful_submissions += 1
        
        # Small delay to avoid overwhelming the service
        time.sleep(0.1)
        
    except Exception as e:
        print(f"✗ Failed to submit job for class {class_id}: {e}")
        failed_submissions += 1
        
        # If too many failures, stop
        if failed_submissions > 5:
            print(f"Too many failures ({failed_submissions}), stopping...")
            break

print(f"\nSummary:")
print(f"Successfully submitted: {successful_submissions}")
print(f"Failed submissions: {failed_submissions}")