# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

from dataclasses import dataclass


@dataclass
class TrainArgs:
    root_path = "/root/sx/sx_pytorch_new/Graphormer/"
    dataset_name = "zinc"
    num_classes = 1
    max_nodes = 128
    dataset_source = "pyg"

    # num_atoms = 512 * 9
    # num_edges = 512 * 3
    # num_in_degree = 512
    # num_out_degree = 512
    # num_spatial = 512
    # num_edge_dis = 128
    # multi_hop_max_dist = 5
    # spatial_pos_max = 1024
    # edge_type = "multi_hop"

    num_atoms = 23
    num_edges = 6
    num_in_degree = 6
    num_out_degree = 6
    num_spatial = 24
    num_edge_dis = 7
    multi_hop_max_dist = 5
    spatial_pos_max = 1024
    edge_type = "multi_hop"

    seed = 1
    pretrained_model_name = "none"
    load_pretrained_model_output_layer = False
    train_epoch_shuffle = False
    user_data_dir = ""

    # with flag
    flag = False
    flag_m = 3  # number of iterations to optimize the perturbations with flag objectives
    flag_step_size = 1e-3  # learning rate of iterations to optimize the perturbations with flag objective
    flag_mag = 1e-3  # magnitude bound for perturbations in flag objectives

    # graphormer_slim
    dropout = 0.0
    attention_dropout = 0.1
    act_dropout = 0.1
    encoder_ffn_embed_dim = 80
    encoder_layers = 12
    encoder_attention_heads = 8
    encoder_embed_dim = 80
    activation_fn = "gelu"

    share_encoder_input_output_embed = False
    encoder_learned_pos = False
    no_token_positional_embeddings = False
    max_positions = None
    apply_graphormer_init = False
    encoder_normalize_before = True  # Graphormer中在encoder前进行LayerNorm
    pre_layernorm = False

    max_steps = 400000
    max_epochs = 10000
    lr = 2e-4
    batch_size = 256
    warmup_steps = 40000
    eps = 1e-8
    beta1 = 0.9
    beta2 = 0.999
    clip_norm = 5.0
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    no_cuda = False
    fp16 = False
    logging_step = 100
    save_path = "./weight"
    pth_name = "zinc_test.pth"
