from polytune import DiscLMTrainer
from polytune.utils import tie_weights
from tokenizers import BertWordPieceTokenizer
from transformers import (AlbertConfig, AlbertForMaskedLM, BertConfig,
                          BertForTokenClassification)

tokenizer = BertWordPieceTokenizer('experiments/electra_small/bert-base-uncased-vocab.txt')

generator_config = AlbertConfig(vocab_size=tokenizer._tokenizer.get_vocab_size(),
                                hidden_size=256,
                                embedding_size=256,
                                num_hidden_layers=3,
                                num_attention_heads=1,
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
                        save_path='electra-small', batch_size=64 * 2, accumulate_grad_batches=1, 
                        weight_decay=0.01,
                        num_workers=2, warmup_steps=10000, learning_rate=5e-4, adam_epsilon=1e-6)
trainer.fit()
