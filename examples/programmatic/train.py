import torch
from lm_saes.config import LanguageModelSAETrainingConfig
from lm_saes.runner import language_model_sae_runner


cfg = LanguageModelSAETrainingConfig.from_flattened(dict(
    # LanguageModelConfig
    model_name = "gpt2",                            # The model name or path for the pre-trained model.
    d_model = 768,                                  # The hidden size of the model.

    # TextDatasetConfig
    dataset_path = 'Skylion007/OpenWebText',                   # The corpus name or path. Each of a data record should contain (and may only contain) a "text" field.
    is_dataset_tokenized = False,                   # Whether the dataset is tokenized.
    is_dataset_on_disk = True,                      # Whether the dataset is on disk. If not on disk, `datasets.load_dataset`` will be used to load the dataset, and the train split will be used for training.
    concat_tokens = True,                     # Whether to concatenate tokens into a single sequence. If False, only data record with length of non-padding tokens larger than `context_size` will be used.
    context_size = 1024,                             # The sequence length of the text dataset.
    store_batch_size = 20,                          # The batch size for loading the corpus.

    # ActivationStoreConfig
    hook_points = ['blocks.8.hook_resid_pre'],        # Hook points to store activations from, i.e. the layer output of which is used for training/evaluating the dictionary. Will run until the last hook point in the list, so make sure to order them correctly.
    use_cached_activations = False,                 # Whether to use cached activations. Caching activation is now not recommended, as it may consume extremely large disk space. (May be tens of TBs for corpus like `openwebtext`)
    n_tokens_in_buffer = 500_000,                   # The number of tokens to store in the activation buffer. The buffer is used to shuffle the activations before training the dictionary.
    
    # SAEConfig
    hook_point_in = 'blocks.8.hook_resid_pre',
    hook_point_out = 'blocks.8.hook_resid_pre',
    use_decoder_bias = True,                        # Whether to use decoder bias.
    expansion_factor = 128,                          # The expansion factor of the dictionary. d_sae = expansion_factor * d_model.
    norm_activation = "token-wise",                 # The normalization method for the activations. Can be "token-wise", "batch-wise" or "none".
    decoder_exactly_fixed_norm = False,              # Whether to enforce the decoder to have exactly unit norm. If False, the decoder will have less than or equal to unit norm.
    use_glu_encoder = False,                        # Whether to use the Gated Linear Unit (GLU) for the encoder.
    l1_coefficient = 2e-4,                        # The L1 regularization coefficient for the feature activations.
    l1_coefficient_warmup_steps = 10000,               # The number of warm-up steps for the L1 regularization coefficient.
    lp = 1,                                         # The p-norm to use for the L1 regularization.
    use_ghost_grads = False,                         # Whether to use the ghost gradients for saving dead features.
    init_decoder_norm = None,                       # The initial norm of the decoder. If None, the decoder will be initialized automatically with the lowest MSE.
    init_encoder_with_decoder_transpose = True,
    apply_decoder_bias_to_pre_encoder = True,
    sparsity_include_decoder_norm = True,

    # LanguageModelSAETrainingConfig
    total_training_tokens = 100_000_000,          # The total number of tokens to train the dictionary.
    lr = 1e-4,                                      # The learning rate for the dictionary training.
    betas = (0.9, 0.9999),                            # The betas for the Adam optimizer.

    lr_scheduler_name = "constantwithwarmup",       # The learning rate scheduler name. Can be "constant", "constantwithwarmup", "linearwarmupdecay", "cosineannealing", "cosineannealingwarmup" or "exponentialwarmup".
    lr_warm_up_steps = 2000,                        # The number of warm-up steps for the learning rate.
    lr_cool_down_steps = 4000,                      # The number of cool-down steps for the learning rate. Currently only used for the "constantwithwarmup" scheduler.
    clip_grad_norm = 0.0,                           # The maximum gradient norm for clipping. If 0.0, no gradient clipping will be performed.
    train_batch_size = 4096,                        # The batch size for training the dictionary, i.e. the number of token activations in a batch.
    feature_sampling_window = 1000,                 # The window size for sampling the feature activations.
    dead_feature_window = 5000,                     # The window size for detecting the dead features.
    dead_feature_threshold = 1e-6,                  # The threshold for detecting the dead features.
    eval_frequency = 1000,                          # The step frequency for evaluating the dictionary.
    log_frequency = 100,                            # The step frequency for logging the training information (to wandb).
    n_checkpoints = 10,                             # The number of checkpoints to save during the training.
    remove_gradient_parallel_to_decoder_directions = False,


    # WandbConfig
    log_to_wandb = True,                            # Whether to log the training information to wandb.
    wandb_project= "test",                      # The wandb project name.

    # RunnerConfig
    device = "cuda",                                # The device to place all torch tensors.
    seed = 42,                                      # The random seed.
    dtype = torch.float32,                          # The torch data type of non-integer tensors.

    exp_name = f"test",                              # The experiment name. Would be used for creating exp folder (which may contain checkpoints and analysis results) and setting wandb run name.
    exp_series = "test",
    exp_result_dir = "results"
))

sparse_autoencoder = language_model_sae_runner(cfg)