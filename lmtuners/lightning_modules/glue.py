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

MODEL_CLASSES = {}  # TODO


class GlueModuleConfig(Namespace):
    def __init__(self):
        super().__init__()


class GlueModule(pl.LightningModule):
    def __init__(self, config):
        self.hparams = config
        self.config = config

        self._init_model()

    def _init_model(self):
        self.config.model_type = self.config.model_type.lower()

        config_class, model_class, tokenizer_class = MODEL_CLASSES[
            self.config.model_type]

        if self.config.tokenizer_type:
            _, _, tokenizer_class = MODEL_CLASSES[
                self.config.tokenizer_type.lower()]

        processor = processors[self.config.task_name]()
        label_list = processor.get_labels()
        self.config.model_type = len(label_list)

        config = config_class.from_pretrained(
            self.config.config_name
            if self.config.config_name else self.config.model_name_or_path,
            self.config.model_type=self.config.model_type,
            finetuning_task=self.config.task_name,
            cache_dir=self.config.cache_dir if self.config.cache_dir else None,
        )
        self.tokenizer = tokenizer_class.from_pretrained(
            self.config.tokenizer_name
            if self.config.tokenizer_name else self.config.model_name_or_path,
            do_lower_case=self.config.do_lower_case,
            cache_dir=self.config.cache_dir if self.config.cache_dir else None,
        )
        self.model = model_class.from_pretrained(
            self.config.model_name_or_path,
            from_tf=bool(".ckpt" in self.config.model_name_or_path),
            config=config,
            cache_dir=self.config.cache_dir if self.config.cache_dir else None,
        )

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3]
        }
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if
                self.config.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

        outputs = self.forward(inputs)
        loss = outputs[0]
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        outputs = model(**inputs)
        eval_loss, logits = outputs[:2]
        return {
            'val_loss': eval_loss,
            'logits': logits,
            'labels': inputs['labels']
        }

    def validation_end(self, outputs):
        avg_val_loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        logits = torch.stack([o['logits'] for o in outputs])
        labels = torch.stack([o['labels'] for o in outputs])

        if self.output_mode == 'classification':
            preds = logits.argmax(dim=-1)
        elif self.output_mode == 'regression':
            preds = logits.squeeze()

        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        result = compute_metrics(self.config.task_name, preds, labels)

        return {
            'tensorboard_logs': result, 'avg_val_loss': avg_val_loss
        }

    def _get_dataloader(self, task, mode):
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
        loader = torch.utils.data.DataLoader(dataset, shuffle=(mode == 'train'), batch_size=self.config.batch_size)
        return loader

    @pl.data_loader
    def train_dataloader(self):
        return self._get_dataloader(self.config.task_name, 'train')

    @pl.data_loader
    def val_dataloader(self):
        eval_task_names = ("mnli", "mnli-mm") if self.config.task_name == "mnli" else (self.config.task_name,)
        loaders = [self._get_dataloader(task, 'dev') for task in eval_task_names]
        return loaders

    @pl.data_loader
    def test_dataloader(self):
        # TODO
        pass
