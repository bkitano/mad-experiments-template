# Model registry — add your model here so configs can reference it by type name.
#
# To register a new model:
#   1. Create models/your_model.py with a class following the standard interface
#   2. Import and add it to MODEL_REGISTRY below
#
# The training script uses MODEL_REGISTRY to look up model classes by the
# "type" field in the config YAML.

from models.deltanet import GroupDeltaNet
from models.transformer import GroupTransformer

MODEL_REGISTRY: dict[str, type] = {
    "GroupDeltaNet": GroupDeltaNet,
    "GroupTransformer": GroupTransformer,
}
