from mlclient import ml_client

def print_registered_assets():

    print("\n📦 Registered Datasets:")
    for data in ml_client.data.list():
        print(f"- {data.name}:{data.version}  ({data.path})")

    print("\n🐳 Registered Environments:")
    for env in ml_client.environments.list():
        print(f"- {env.name}:{env.version}")

    print("\n💻 Compute Targets:")
    for comp in ml_client.compute.list():
        print(f"- {comp.name} [{comp.type}]")

if __name__ == "__main__":
    print_registered_assets()
