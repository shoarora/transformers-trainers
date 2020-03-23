import os

import fire
from pytorch_lightning import Trainer
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from transformers import AlbertConfig, AlbertForMaskedLM, BertConfig, BertForMaskedLM

from lmtuners import DiscLMTrainingModule, DiscLMTrainingModuleConfig
from lmtuners.datasets import PreTokenizedCollater, create_pretokenized_dataset
from lmtuners.models import AlbertForTokenClassification
from lmtuners.utils import tie_weights


def main(tokenizer_path,
         dataset_path,
         save_path='alectra-small',
         max_steps=1e6,
         accumulate_grad_batches=1,
         gpus=None,
         num_tpu_cores=None,
         distributed_backend=None,
         val_check_interval=0.25,
         val_check_percent=0.25,
         generator_type='albert',
         num_hidden_groups=1,
         mlm_prob=0.15,
         learning_rate=5e-4,
         warmup_steps=10000,
         batch_size=128,
         num_workers=2,
         shuffle=True,
         resume_from_checkpoint=None,
         use_polyaxon=False):
    # init tokenizer.  only need it for the special chars.
    tokenizer = BertWordPieceTokenizer(tokenizer_path)

    # init generator.
    if generator_type == 'albert':
        generator_config = AlbertConfig(
            vocab_size=tokenizer._tokenizer.get_vocab_size(),
            hidden_size=256,
            embedding_size=128,
            num_hidden_layers=3,
            num_attention_heads=1,
            num_hidden_groups=num_hidden_groups,
            intermediate_size=1024,
            max_position_embeddings=128)
        generator = AlbertForMaskedLM(generator_config)
    elif generator_type == 'bert':
        generator_config = BertConfig(
            vocab_size=tokenizer._tokenizer.get_vocab_size(),
            hidden_size=128,
            num_hidden_layers=3,
            num_attention_heads=1,
            intermediate_size=256,
            max_position_embeddings=128)
        generator = BertForMaskedLM(generator_config)
        tie_weights(generator.cls.predictions.decoder, generator.bert.embeddings.word_embeddings)
    else:
        raise Exception(f"invalid generator type: {generator_type}")

    # init discriminator.
    discriminator_config = AlbertConfig(
        vocab_size=tokenizer._tokenizer.get_vocab_size(),
        hidden_size=256,
        embedding_size=128,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_hidden_groups=num_hidden_groups,
        intermediate_size=1024,
        max_position_embeddings=128)
    discriminator = AlbertForTokenClassification(discriminator_config)

    # tie the embeddingg weights.
    tie_weights(discriminator.base_model.embeddings.word_embeddings,
                generator.base_model.embeddings.word_embeddings)
    tie_weights(discriminator.base_model.embeddings.position_embeddings,
                generator.base_model.embeddings.position_embeddings)
    tie_weights(discriminator.base_model.embeddings.token_type_embeddings,
                generator.base_model.embeddings.token_type_embeddings)

    if generator_type == 'albert':
        discriminator.albert.encoder.albert_layer_groups = generator.albert.encoder.albert_layer_groups

    # init training module.
    training_config = DiscLMTrainingModuleConfig(max_steps,
                                                 save_path=save_path,
                                                 weight_decay=0.01,
                                                 learning_rate=learning_rate,
                                                 epsilon=1e-6,
                                                 warmup_steps=warmup_steps)
    if use_polyaxon:
        checkpoint_fn = polyaxon_checkpoint_fn
    else:
        checkpoint_fn = None
    lightning_module = DiscLMTrainingModule(generator,
                                            discriminator,
                                            training_config,
                                            checkpoint_fn=checkpoint_fn)

    # init trainer.
    trainer = Trainer(accumulate_grad_batches=accumulate_grad_batches,
                      gpus=gpus,
                      num_tpu_cores=num_tpu_cores,
                      distributed_backend=distributed_backend,
                      max_steps=max_steps,
                      resume_from_checkpoint=resume_from_checkpoint,
                      val_check_percent=val_check_percent,
                      val_check_interval=val_check_interval)

    # init dataloaders.
    train_loader, val_loader, _ = get_dataloaders(tokenizer, dataset_path,
                                                  trainer, mlm_prob,
                                                  batch_size, num_workers,
                                                  shuffle)

    # train.
    trainer.fit(lightning_module, train_loader, val_loader)

    # save the model.
    output_path = os.path.join(save_path, 'discriminator', 'final')
    os.makedirs(output_path, exist_ok=True)
    lightning_module.discriminator.base_model.save_pretrained(output_path)
    if checkpoint_fn:
        checkpoint_fn(lightning_module)


def polyaxon_checkpoint_fn(lightning_module):
    from polyaxon_client.tracking import Experiment
    exp = Experiment()
    exp.outputs_store.upload_dir(lightning_module.config.save_path)
    exp.outputs_store.upload_dir('lightning_logs')


def get_dataloaders(tokenizer, dataset_path, trainer, mlm_prob, batch_size,
                    num_workers, shuffle):
    def get_dataloader(path):
        paths = [os.path.join(path, name) for name in os.listdir(path)]
        dataset = create_pretokenized_dataset(paths)

        collater = PreTokenizedCollater(
            mlm=True,
            mlm_prob=mlm_prob,
            pad_token_id=tokenizer.token_to_id("[PAD]"),
            mask_token_id=tokenizer.token_to_id("[MASK]"),
            vocab_size=tokenizer._tokenizer.get_vocab_size(),
            rand_replace=False)

        return DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          collate_fn=collater,
                          shuffle=shuffle)

    train_loader = get_dataloader(os.path.join(dataset_path, 'train'))
    val_loader = get_dataloader(os.path.join(dataset_path, 'val'))
    test_loader = get_dataloader(os.path.join(dataset_path, 'test'))
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    fire.Fire(main)
