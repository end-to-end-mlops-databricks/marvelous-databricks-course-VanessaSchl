"""Configuration module for the project."""

from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Class to store the configuration for the project."""

    num_features: List[str]
    cat_features: List[str]
    original_target: str
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    ab_test: Dict[str, Any]  # Dictionary to hold A/B test parameters

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)