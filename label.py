import logging
from typing import Union
from pydantic import BaseModel
import yaml

logger = logging.getLogger(__name__)

def parse_labels(file: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    if file is None:
        raise ValueError("File path cannot be None")
    else:
        logger.debug(f"Opening file: {file}")
        try:
            with open(file, 'r') as content:
                if file.endswith(".yaml") or file.endswith(".yml"):
                    logger.debug("Processing as a YAML file.")
                    for id, name in yaml.safe_load(content)["names"].items():
                        labels[int(id)] = name
                else:
                    logger.debug("Processing as a text file.")
                    for line in content:
                        id, name = line.strip().split(' ', maxsplit=1)
                        labels[int(id)] = name.strip()
        except Exception as e:
            logger.error(f"An error occurred while processing the file: {e}")
            raise e
    return labels