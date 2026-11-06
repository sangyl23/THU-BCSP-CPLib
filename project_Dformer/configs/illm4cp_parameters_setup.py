
def add_model_args(parser):
    # model parameters
    parser.add_argument('--dropout', type = float, default = 0.08)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_mixing',
                        choices = ['Channel_independent', 'Channel_mixing', 'RB_independent', 'M_independent'])
    parser.add_argument('--instance_norm', type = str, default = 'norm_max',
                        choices = ['norm_2', 'norm_max', 'RevIN', 'wo'])
    parser.add_argument('--ifAD', type = eval, default = True)
    
    # Transformer parameters
    parser.add_argument('--task_name', type = str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--enc_in', type = int, default = 512, help = 'encoder input size')
    parser.add_argument('--dec_in', type = int, default = 512, help = 'decoder input size')
    parser.add_argument('--c_out', type = int, default = 512, help = 'output size')
    parser.add_argument('--d_model', type = int, default = 128, help = 'dimension of CSI front-end and decoder')
    parser.add_argument('--n_heads', type = int, default = 8, help = 'num of heads')
    parser.add_argument('--e_layers', type = int, default = 2, help = 'num of encoder layers')
    parser.add_argument('--d_layers', type = int, default = 1, help = 'num of decoder layers')
    parser.add_argument('--d_ff', type = int, default = 64, help ='dimension of fcn')
    parser.add_argument('--factor', type = int, default = 3, help ='probsparse attn factor')
    parser.add_argument('--distil', action = 'store_false', help = 'whether to use distilling in encoder', default = True)
    parser.add_argument('--embed', type = str, default = 'timeF', help = 'time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type = str, default = 'gelu', help = 'activation')
    
    # Embedding parameters
    parser.add_argument('--ifemb_conditions', type = eval, default = False)
    parser.add_argument('--condition_embedding', type = str, default = 'frequency_pe',
                        choices = ['frequency_pe', 'continuous', 'separate', 'concat', 'frequency_pe_wo_flim'])
    parser.add_argument('--num_speed_conditions', type = int, default = 8)
    parser.add_argument('--num_interval_conditions', type = int, default = 8)
    
    # LLM4CP-specific parameters
    parser.add_argument('--ifuse_gpt2', type = eval, default = True)
    parser.add_argument('--gpt_type', type = str, default = 'gpt2',
                        choices = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--gpt_layers', type = int, default = 6)
    parser.add_argument('--ifuse_seblock', type = eval, default = True)
    parser.add_argument('--se_layers', type = int, default = 4)
    parser.add_argument('--se_dim', type = int, default = 64)
    parser.add_argument('--patch_size', type = int, default = 4)
    parser.add_argument('--gpt2_lr', type = float, default = 1e-5)
    parser.add_argument('--ifgpt2_lr_warmup_clipping', type = eval, default = True)
    parser.add_argument('--gpt2_warmup_ratio', type = float, default = 0.05)
    parser.add_argument('--max_grad_norm', type = float, default = 1.0)

    parser.add_argument('--iffinetune_attention', type = eval, default = False)
    parser.add_argument('--iffinetune_mlp', type = eval, default = False)
    parser.add_argument('--iffinetune_ln', type = eval, default = True)
    parser.add_argument('--iflora', type = eval, default = True)
    parser.add_argument('--lora_rank', type = int, default = 8)
    parser.add_argument('--lora_alpha', type = float, default = 16.0)
    parser.add_argument('--lora_dropout', type = float, default = 0.05)
    parser.add_argument('--ifcausal_attention', type = eval, default = False)
    parser.add_argument('--ifres_prediction', type = eval, default = False)
    
    # Dformer-specific parameters
    parser.add_argument('--emb_module', type = str, default = 'temporal_emb',
                        choices = ['variable_emb', 'temporal_emb'])
    parser.add_argument('--embed_learning', type = str, default = 'time',
                        choices = ['time', 'doppler'])
    parser.add_argument('--doppler_keep_ratio', type = float, default = 0.8)
    
    return parser
