

import os
from pathlib import Path

from safetensors.torch import load_file, save_file

from .model_persister import ModelPersister


class SafetensorModelPersister(ModelPersister):
    def __init__(self, *args, **kwargs):
        super(SafetensorModelPersister, self).__init__(*args, **kwargs)

    def model_persist_exist(self, layer_name, saving_path):
        safetensor_exists = os.path.exists(str(saving_path / (layer_name + "safetensors")))
        done_marker_exists = os.path.exists(str(saving_path / (layer_name + "safetensors.done")))
        return safetensor_exists and done_marker_exists

    def persist_model(self, state_dict, layer_name, saving_path):
        save_file(state_dict, saving_path / (layer_name + "safetensors"))
        print(f"saved as: {saving_path / (layer_name + 'safetensors')}")
        # set done marker
        (saving_path / (layer_name + "safetensors.done")).touch()

    def load_model(self, layer_name, path):
        base = Path(path)

        # Possibili pattern:
        # 1) layer_name + "safetensors"      -> es: "model.embed_tokens" + "safetensors"
        # 2) layer_name + ".safetensors"     -> es: "model.embed_tokens." + ".safetensors"
        candidates = [
            base / (layer_name + "safetensors"),
            base / ("lm_head" + ".safetensors"),
        ]

        for fname in candidates:
            if fname.exists():
                return load_file(fname, device="cpu")

        raise FileNotFoundError(
            f"None of the expected files for layer '{layer_name}' found in {base}:\n"
            + "\n".join(str(c) for c in candidates)
        )