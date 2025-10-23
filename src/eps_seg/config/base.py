from typing import Optional
from pydantic import BaseModel, Field
import yaml
from pathlib import Path

class BaseEPSConfig(BaseModel):
    config_yaml_path: Optional[Path] = Field(default=None, description="Path to the YAML configuration file")

    @classmethod
    def from_yaml(cls, yaml_path: Path):
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        obj = cls(**config_dict)
        obj.config_yaml_path = yaml_path
        return obj