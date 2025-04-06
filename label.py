import logging
from typing import Union
from pydantic import BaseModel
import yaml

logger = logging.getLogger(__name__)

class Label(BaseModel):
    id: int
    name: Union[str, None]

def parse_labels(file: Union[str, None] = None):
    labels = []
    if file is None:
       logger.warning("No file provided, generating labels without names.")
       labels = [Label(i, None) for i in range(1000)]
    else:
        logger.debug(f"Opening file: {file}")
        try:
            with open(file, 'r') as content:
                if file.endswith(".yaml") or file.endswith(".yml"):
                        logger.debug("Processing as a YAML file.")
                        for id, name in yaml.safe_load(content)["names"].items():
                            labels.append(Label(id=int(id), name=name.strip()))
                else:
                    logger.debug("Processing as a text file.")
                    for line in content:
                        id, name = line.strip().split(' ', maxsplit=1)
                        detection = Label(id=int(id), name=name.strip())
                        labels.append(detection)
        except Exception as e:
            logger.error(f"An error occurred while processing the file: {e}")
            raise e
    return {label.id: label for label in labels}