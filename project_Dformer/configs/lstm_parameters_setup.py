
def add_model_args(parser):
    # model parameters
    parser.add_argument('--dropout', type = float, default = 0.08)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_independent',
                        choices = ['Channel_independent', 'Channel_mixing', 'RB_independent', 'M_independent'])
    parser.add_argument('--instance_norm', type = str, default = 'norm_max',
                        choices = ['norm_2', 'norm_max', 'RevIN', 'wo'])
    parser.add_argument('--ifAD', type = eval, default = True)
    
    # lstm parameters
    parser.add_argument('--task_name', type = str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--input_size', type = int, default = 64, help = 'lstm input size')
    parser.add_argument('--hidden_size', type = int, default = 64, help = 'lstm hidden size')
    parser.add_argument('--features', type = int, default = 2, help = 'model input size')
    parser.add_argument('--num_layers', type = int, default = 3, help = 'num of lstm layers')
    
    return parser
