"""This script handles the deployment of a hotel reservation prediction model to a Databricks serving endpoint."""

import argparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput
from hotel_reservations.config import ProjectConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()

config_path = f"{args.root_path}/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path)

model_version = dbutils.jobs.taskValues.get(
    taskKey="evaluate_model", key="model_version"
)

workspace = WorkspaceClient()

workspace.serving_endpoints.update_config_and_wait(
    name="hotel-reservations-model-serving-vs",
    served_entities=[
        ServedEntityInput(
            entity_name=f"{config.catalog_name}.{config.schema_name}.vs_hotel_reservations_model_basic",
            scale_to_zero_enabled=True,
            workload_size="Small",
            entity_version=model_version,
        )
    ],
)
