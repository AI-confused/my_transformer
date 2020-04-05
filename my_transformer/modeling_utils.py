"""PyTorch BERT model."""

import logging
import os
import typing

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from .file_utils import (
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
)

logger = logging.getLogger(__name__)

class PreTrainedModel(nn.Module):
    # class attributes
    config_class = None
    base_model_prefix = ""
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                    )
            )
        #save config in model
        self.config = config

    def get_output_embeddings(self):
        return None
        
    def tie_weight(self):
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            pass

    def init_weights(self):
        # init weights
        self.apply(self._init_weights)

        # prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        # tie weights if needed
        #self.tie_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        from_tf = kwargs.pop("from_tf", None)
        output_loading_info = kwargs.pop("output_loading_info", False)

        #load config
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                **kwargs,
            )
        #load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    #load from TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnviromentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                            )
                    )
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("pretrained_model_name_or_path is not dir")
        else:
            archive_file = None

        # Instantiate model
        model = cls(config, *model_args, **kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(archive_file, map_location='cpu')
            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                    "If you tried to load a PyTorch model from a TF checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if archive_file.endswith(".index"):
                # load from tf 1.0 checkpoint
                model = cls.load_tf_weights(model, config, archive_file[:-6]) # remove '.index'
            else:
                # load from tf 2.0
                pass
        else:
            # load from pytorch
            # convert old format to new format if needed
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            # apply load recursively
            def load(module: nn.Module, prefix=""):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            start_prefix = ""
            model_to_load = model
            if not hasattr(model, cls.base_model_prefix) and any(
                    s.startswith(cls.base_model_prefix) for s in state_dict.keys()
            ):
                start_prefix = cls.base_model_prefix + '.'
            if hasattr(model, cls.base_model_prefix) and not any(
                    s.startswith(cls.base_model_prefix) for s in state_dict.keys()
            ):
                model_to_load = getattr(model, cls.base_model_prefix)

            load(model_to_load, prefix=start_prefix)

            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + '.')[-1] for key in model.state_dict().keys()
                ]

                missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

            if len(missing_keys) > 0:
                logger.info(
                    "Weights of {} not initialized from pretrained model: {}".format(
                        model.__class__.__name__, missing_keys
                    )
                )

            if len(unexpected_keys) > 0:
                logger.info(
                    "Weights from pretrained model not used in {}: {}".format(
                        model.__class__.__name__, unexpected_keys
                    )
                )

            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )

        #model.tie_weight() # make sure token embedding weights are still tied if needed

        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model
        

            

            
                
                
                
            
