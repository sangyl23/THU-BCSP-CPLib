
def add_model_args(parser):
    # model parameters
    parser.add_argument('--dropout', type = float, default = 0.08)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_independent',
                        choices = ['Channel_independent', 'Channel_mixing', 'RB_independent', 'M_independent'])
    parser.add_argument('--instance_norm', type = str, default = 'norm_max',
                        choices = ['norm_2', 'norm_max', 'RevIN', 'wo'])
    parser.add_argument('--ifAD', type = eval, default = False)
    
    # Transformer parameters
    parser.add_argument('--task_name', type = str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--enc_in', type = int, default = 2, help = 'encoder input size')
    parser.add_argument('--dec_in', type = int, default = 2, help = 'decoder input size')
    parser.add_argument('--c_out', type = int, default = 2, help = 'output size')
    parser.add_argument('--d_model', type = int, default = 768, help = 'dimension of model')
    parser.add_argument('--n_heads', type = int, default = 8, help = 'num of heads')
    parser.add_argument('--e_layers', type = int, default = 2, help = 'num of encoder layers')
    parser.add_argument('--d_layers', type = int, default = 1, help = 'num of decoder layers')
    parser.add_argument('--d_ff', type = int, default = 64, help ='dimension of fcn')
    parser.add_argument('--factor', type = int, default = 3, help ='probsparse attn factor')
    parser.add_argument('--distil', action = 'store_false', help = 'whether to use distilling in encoder', default = True)
    parser.add_argument('--embed', type = str, default = 'timeF', help = 'time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type = str, default = 'gelu', help = 'activation')
    
    # Embedding parameters
    parser.add_argument('--ifemb_conditions', type = eval, default = True)
    parser.add_argument('--condition_embedding', type = str, default = 'frequency_pe',
                        choices = ['frequency_pe', 'continuous', 'separate', 'concat'])
    parser.add_argument('--num_speed_conditions', type = int, default = 8)
    parser.add_argument('--num_interval_conditions', type = int, default = 8)
    
    # LLM4CP-specific parameters
    parser.add_argument('--gpt_type', type = str, default = 'gpt2',
                        choices = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--gpt_layers', type = int, default = 3)
    parser.add_argument('--iffinetune_ln', type = eval, default = True)
    parser.add_argument('--iffinetune_mlp', type = eval, default = True)
    parser.add_argument('--res_layers', type = int, default = 4)
    parser.add_argument('--res_dim', type = int, default = 64)
    parser.add_argument('--patch_size', type = int, default = 4)
    
    return parser
