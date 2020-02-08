from polytune import DiscLMTrainer
from polytune.utils import tie_weights
from tokenizers import BertWordPieceTokenizer
from transformers import (AlbertConfig, AlbertForMaskedLM, BertConfig,
                          BertForTokenClassification)

tokenizer = BertWordPieceTokenizer('data/mercari-wordpiece-vocab.txt')

generator_config = AlbertConfig(vocab_size=tokenizer._tokenizer.get_vocab_size(),
                                hidden_size=256,
                                num_hidden_layers=2,
                                num_attention_heads=2,
                                intermediate_size=256,
                                max_position_embedding=128)
discriminator_config = BertConfig(vocab_size=tokenizer._tokenizer.get_vocab_size(),
                                  hidden_size=256,
                                  num_hidden_layers=12,
                                  num_attention_heads=4,
                                  intermediate_size=1024,
                                  max_position_embedding=128)
generator = AlbertForMaskedLM(generator_config)
discriminator = BertForTokenClassification(discriminator_config)


tie_weights(generator.albert.embeddings.word_embeddings, discriminator.bert.embeddings.word_embeddings)
tie_weights(generator.albert.embeddings.position_embeddings, discriminator.bert.embeddings.position_embeddings)
tie_weights(generator.albert.embeddings.token_type_embeddings, discriminator.bert.embeddings.token_type_embeddings)


trainer = DiscLMTrainer(generator, discriminator, tokenizer, 'experiments/electra_small/data',
                        save_path='mercari-electra', batch_size=64 * 4, accumulate_grad_batches=1, num_workers=4)
trainer.fit()
