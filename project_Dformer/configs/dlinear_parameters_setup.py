
def add_model_args(parser):
    # model parameters
    parser.add_argument('--dropout', type = float, default = 0.)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_independent2',
                        choices = ['Channel_independent1', 'Channel_independent2', 'Channel_mixing'])
    parser.add_argument('--instance_norm', type = str, default = 'wo',
                        choices = ['norm_2', 'norm_max', 'RevIN', 'revin', 'wo'])
    parser.add_argument('--enc_in', type = int, default = 2, help = 'encoder input size')
    parser.add_argument('--d_model', type = int, default = 64, help = 'dimension of model')
    parser.add_argument('--embed', type = str, default = 'fixed', help = 'time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--pe_type', type = str, default = 'grouped_time',
                        choices = ['grouped_time', 'time', 'position', 'wo_pos', 'wo_emb'])      
    parser.add_argument('--task_name', type = str, default = 'long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--omega0', type = float, default = 1.)
    parser.add_argument('--omega_model', type = str, default = 'wo',
                        choices = ['cnngru', 'dcnn', 'wo'])
    
    ## DLinear-specific parameters
    parser.add_argument('--individual', type = eval, default = False)
    parser.add_argument('--moving_avg', type = int, default = 25, help='window size of moving average')
    
    return parser
