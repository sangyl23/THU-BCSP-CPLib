
def add_model_args(parser):
    # model parameters
    parser.add_argument('--dropout', type = float, default = 0.08)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_independent',
                        choices = ['Channel_independent', 'Channel_mixing', 'RB_independent', 'M_independent'])
    parser.add_argument('--instance_norm', type = str, default = 'norm_max',
                        choices = ['norm_2', 'norm_max', 'RevIN', 'wo'])
    parser.add_argument('--ifAD', type = eval, default = True)
    
    # Transformer parameters
    parser.add_argument('--task_name', type = str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--enc_in', type = int, default = 2, help = 'encoder input size')
    parser.add_argument('--dec_in', type = int, default = 2, help = 'decoder input size')
    parser.add_argument('--c_out', type = int, default = 2, help = 'output size')
    parser.add_argument('--d_model', type = int, default = 128, help = 'dimension of model')
    parser.add_argument('--n_heads', type = int, default = 8, help = 'num of heads')
    parser.add_argument('--e_layers', type = int, default = 2, help = 'num of encoder layers')
    parser.add_argument('--d_layers', type = int, default = 1, help = 'num of decoder layers')
    parser.add_argument('--d_ff', type = int, default = 64, help ='dimension of fcn')
    parser.add_argument('--factor', type = int, default = 3, help ='probsparse attn factor')
    parser.add_argument('--distil', action = 'store_false', help = 'whether to use distilling in encoder', default = True)
    parser.add_argument('--embed', type = str, default = 'fixed', help = 'time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type = str, default = 'gelu', help = 'activation')
    
    # Embedding parameters
    parser.add_argument('--ifemb_conditions', type = eval, default = False)
    parser.add_argument('--condition_embedding', type = str, default = 'frequency_pe',
                        choices = ['frequency_pe', 'continuous', 'separate', 'concat', 'frequency_pe_wo_flim'])
    parser.add_argument('--num_speed_conditions', type = int, default = 8)
    parser.add_argument('--num_interval_conditions', type = int, default = 8)
    
    ## FEDformer-specific parameters
    parser.add_argument('--version', type = str, default = 'Fourier',
                        choices = ['Fourier', 'Wavelets'])
    parser.add_argument('--mode_select', type = str, default = 'random',
                        choices = ['random', 'low'])
    parser.add_argument('--modes', type = int, default = 32, help = 'selected frequency number')
    parser.add_argument('--moving_avg', type = int, default = 25, help='window size of moving average')
    
    return parser
