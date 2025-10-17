from pydantic import BaseModel, Field
import yaml
from pathlib import Path

class BaseEPSConfig(BaseModel):
    
    @classmethod
    def from_yaml(cls, yaml_path: Path):
        print(cls)
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)