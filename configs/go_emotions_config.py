config = {}

config["batch_size"] = 32   
config["epochs"] = 100      
config["no_cuda"] = False   
config["seed"] = 1265
config["log_interval"] = 1  

config["learning_rate"] = 6e-4
config["min_learning_rate"] = 6e-5
config["learning_rate_decay_iters"] = 600000

config["warmup_iters"] = 2000
config["weight_decay"] = 1e-2
config["betas"] = (0.9, 0.95)

config["num_embeddings"] = 768
config["num_heads"] = 12
config["num_layers"] = 12
config["block_size"] = 1024
config["vocab_size"] = 50257
config["pad_token_id"] = 50256

config["num_labels"] = 28
config["dropout"] = 0.1
config["problem_type"] = "multi_label_classification"
config["tokenizer"] = "GPT2Tokenizer"
config["data_set"] = "GOEMOTIONS"
config["max_sequence_len"] = 100
config["model_type"] = "gpt2"