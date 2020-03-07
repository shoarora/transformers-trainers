import logging
import os
from argparse import Namespace

import pytorch_lightning as pl
import torch
from transformers import get_linear_schedule_with_warmup
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features

logger = logging.getLogger(__name__)


class GlueModuleConfig(Namespace):
    def __init__(self):
        super().__init__()


class GlueModule(pl.LightningModule):
    def __init__(self, model, tokenizer, config):
        pass

    def _get_dataloader(self, mode):
        task = self.config.task

        processor = processors[task]()
        output_mode = output_modes[task]
        if mode == 'train':
            examples = processor.get_train_examples(self.config.data_dir)
        elif mode == 'dev':
            examples = processor.get_dev_examples(self.config.data_dir)

        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and self.model.base_model_prefix in [
                "roberta", "xlmroberta"
        ]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=label_list,
            max_length=self.config.max_length,
            output_mode=output_mode,
            pad_on_left=bool(self.model.base_model_prefix in
                             ["xlnet"]),  # pad on the left for xlnet
            pad_token=self.tokenizer.pad_token,
            pad_token_segment_id=4
            if self.model.base_model_prefix in ["xlnet"] else 0,
        )

        # Convert to Tensors and build dataset

        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features],
                                          dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                          dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features],
                                      dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features],
                                      dtype=torch.float)

        dataset = torch.utils.data.TensorDataset(all_input_ids,
                                                 all_attention_mask,
                                                 all_token_type_ids,
                                                 all_labels)
        return dataset

    @pl.data_loader
    def train_dataloader(self):
        return self._get_dataloader('train')

    @pl.data_loader
    def val_dataloader(self):
        return self._get_dataloader('dev')

    @pl.data_loader
    def test_dataloader(self):
        # TODO
        pass
