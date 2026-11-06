
def add_model_args(parser):
    # model parameters
    parser.add_argument('--dropout', type = float, default = 0.08)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_independent2',
                        choices = ['Channel_independent1', 'Channel_independent2', 'Channel_mixing'])
    parser.add_argument('--instance_norm', type = str, default = 'wo',
                        choices = ['norm_2', 'norm_max', 'RevIN', 'revin', 'wo'])
    # Transformer parameters
    parser.add_argument('--task_name', type = str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--enc_in', type = int, default = 2, help = 'encoder input size')
    parser.add_argument('--dec_in', type = int, default = 2, help = 'decoder input size')
    parser.add_argument('--c_out', type = int, default = 2, help = 'output size')
    parser.add_argument('--d_model', type = int, default = 64, help = 'dimension of model')
    parser.add_argument('--n_heads', type = int, default = 8, help = 'num of heads')
    parser.add_argument('--e_layers', type = int, default = 2, help = 'num of encoder layers')
    parser.add_argument('--d_layers', type = int, default = 2, help = 'num of decoder layers')
    parser.add_argument('--d_ff', type = int, default = 64, help ='dimension of fcn')
    parser.add_argument('--factor', type = int, default = 3, help ='probsparse attn factor')
    parser.add_argument('--distil', action = 'store_false', help = 'whether to use distilling in encoder', default = True)
    parser.add_argument('--embed', type = str, default = 'fixed', help = 'time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type = str, default = 'gelu', help = 'activation')
    parser.add_argument('--pe_type', type = str, default = 'grouped_time',
                        choices = ['grouped_time', 'time', 'position', 'wo_pos', 'wo_emb'])
    parser.add_argument('--omega0', type = float, default = 1.)
    parser.add_argument('--omega_model', type = str, default = 'wo',
                        choices = ['cnngru', 'dcnn', 'wo'])
    ## NS_Transformer-specific parameters
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[64, 64],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    
    return parser
