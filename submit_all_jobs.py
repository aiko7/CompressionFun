import time
import argparse
from azure.ai.ml import command, Output, Input
from azure.ai.ml.constants import AssetTypes
from mlclient import ml_client

parser = argparse.ArgumentParser(description="Job sending configuration")
parser.add_argument("--classes", type=int, required=True, help="Number of classes to use from tiny imagenet200")
parser.add_argument("--num_nodes", type=int, required=True, help="Number of parallel nodes/jobs to use, MUST FIT AML CONFIGURATION!")
parser.add_argument("--parallel_models", type=int, required=True, help="Number of models to train in parallel per node")

args = parser.parse_args()

try:
    env = ml_client.environments.get("poetry-env", version="21")
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

total_classes = args.classes
num_nodes = args.num_nodes
classes_per_node = total_classes // num_nodes
remainder = total_classes % num_nodes

print(f"Distributing {total_classes} classes across {num_nodes} nodes:")
print(f"Base classes per node: {classes_per_node}")
print(f"Parallel models per node: {args.parallel_models}")
if remainder > 0:
    print(f"First {remainder} nodes will get 1 additional class")

successful_submissions = 0
failed_submissions = 0

for node_id in range(num_nodes):
    start_class = node_id * classes_per_node + min(node_id, remainder)
    
    if node_id < remainder:
        end_class = start_class + classes_per_node + 1
    else:
        end_class = start_class + classes_per_node
    
    class_list = ",".join(str(i) for i in range(start_class, end_class))
    
    print(f"Node {node_id}: Training classes {start_class}-{end_class-1} ({end_class-start_class} classes) with {args.parallel_models} parallel models")
    
    job = command(
        display_name=f"Autoencoder Batch Node {node_id} (Parallel={args.parallel_models})",
        experiment_name="ae-compression-exp",
        description=f"Training autoencoders on node {node_id} for classes {start_class}-{end_class-1} with {args.parallel_models} parallel models",
        code=".",
        command=(
            "python train.py "
            f"--class_list {class_list} "
            f"--parallel_models {args.parallel_models} "
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
                path=f"azureml://datastores/workspaceblobstore/paths/ae_ckpts/node_{node_id}"
            )
        }
    )

    try:
        out_job = ml_client.jobs.create_or_update(job)
        print(f"Submitted batch job {out_job.name} for node {node_id} (classes {start_class}-{end_class-1})")
        successful_submissions += 1
        
        # delay to avoid overwhelming the service
        time.sleep(0.3)
        
    except Exception as e:
        print(f"Failed to submit batch job for node {node_id}: {e}")
        failed_submissions += 1
        
        # If too many failures, stop
        if failed_submissions > 2:
            print(f"Too many failures ({failed_submissions}), stopping...")
            break

print(f"\nSummary:")
print(f"Successfully submitted: {successful_submissions} batch jobs")
print(f"Failed submissions: {failed_submissions}")
print(f"Each node will train {args.parallel_models} models in parallel")