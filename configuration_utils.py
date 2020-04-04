""" Configuration base class and utilities."""

import copy
import json
import logging
import os
from typing import Dict, Optional, Tuple



logger = logging.getLogger(__name__)

class PretrainedConfig(object):
    model_type = ''

    def __init__(self, **kwargs):
        # Attributes with defaults
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_past = kwargs.pop("output_past", True)  # Not used by all models
        self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.min_length = kwargs.pop("min_length", 0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.id2label = kwargs.pop("id2label", {i: "LABEL_{}".format(i) for i in range(self.num_labels)})
        self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        self.label2id = kwargs.pop("label2id", dict(zip(self.id2label.values(), self.id2label.keys())))
        self.label2id = dict((key, int(value)) for key, value in self.label2id.items())

        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # task specific arguments
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

        @property
        def num_labels(self):
            return self._num_labels

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "Pretrainedconfig":
            config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
            return cls.from_dict(config_dict, **kwargs)

        @classmethod
        def get_config_dict(
                cls, pretrained_model_name_or_path: str, **kwargs
        ) -> Tuple[Dict, Dict]:
            if os.path.isdir(pretrained_model_name_or_path):
                config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
            elif os.path.isfile(pretrained_model_name_or_path):
                config_file = pretrained_model_name_or_path
            else:
                msg = ("can't load '{}'. make sure config dir or file is available.".format(pretrained_model_name_or_path))
                raise EnviromentError(msg)
            try:
                config_dict = cls._dict_from_json_file(config_file)
            except EnviromentError:
                raise EnviromentError("cannot load config dict from {}, please check config file".format(config_file))
            logger.info("loading configuration file {}".format(config_file))
            return config_dict, kwargs

        @classmethod
        def from_dict(cls, config_dict: Dict, **kwargs) -> "Pretrainedconfig":
            return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

            config = cls(**config_dict)

            if hasattr(config, "pruned_heads"):
                pass
            # update config with kwargs if needed
            to_remove = []
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    to_remove.append(key)
            for key in to_remove:
                kwargs.pop(key, None)

            logger.info("Model config %s", str(config))
            if return_unused_kwargs:
                return config, kwargs
            else:
                return config
                
        
        @classmethod
        def from_json_file(cls, json_file: str) -> "Pretrainedconfig":
            config_dict = cls._dict_from_json_file(json_file)
            return cls(**config_dict)
        
        @classmethod
        def _dict_from_json_file(cls, json_file: str):
            with open(json_file, 'r', encoding='utf-8') as reader:
                text = reader.read()
            return json.loads(text)
            
            
            
