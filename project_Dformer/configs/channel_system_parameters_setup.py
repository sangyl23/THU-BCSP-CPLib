
def add_channel_system_args(parser):    
    # channel and system parameters
    parser.add_argument('--channel_type', type = str, default = 'Industrial_Radio',
                        choices = ['QuaDRiGa', 'DeepMIMO', 'Industrial_Radio'])
    # parser.add_argument('--scenario', type = str, default = '3GPP_38.901_UMa_NLOS')
    parser.add_argument('--v', type = str, default = '3p6kmh',
                        choices = ['3p6kmh', '20kmh', '40kmh', '60kmh', '80kmh'])
    parser.add_argument('--interval', type = str, default = '1en3',
                        choices = ['0p5en3', '1en3', '1p5en3', '2en3'])
    parser.add_argument('--snr', type = str, default = '20dB',
                        choices = ['15dB', '20dB', '25dB', '30dB'])
    parser.add_argument('--UE_num', type = int, default = 4)
    parser.add_argument('--RB_num', type = int, default = 8)
    parser.add_argument('--M', type = int, default = 1)
    parser.add_argument('--M_h', type = int, default = 1)
    parser.add_argument('--M_v', type = int, default = 1)
    parser.add_argument('--seq_len', type = int, default = 64)
    parser.add_argument('--pred_len', type = int, default = 8)
    parser.add_argument('--label_len', type = int, default= 32, help = 'start token length of transformer decoder')
    parser.add_argument('--t0', type = int, default = 64)
    parser.add_argument('--read_uplink', type = eval, default = False)
    
    parser.add_argument('--path_train', type = str, default = 'Industrial_Radio/process_v3p6kmh_1en3_his64_pre8/train_ULsnr10dB_DLsnr20dB')
    parser.add_argument('--path_eval', type = str, default = 'Industrial_Radio/process_v3p6kmh_1en3_his64_pre8/test_ULsnr10dB_DLsnr20dB')
    
    
    # grouped port estimation
    parser.add_argument('--port_estimation_mode', type = str, default = 'grouped_cyclic_angle_delay',
                        choices = ['full', 'full_long_period', 
                                   'partial_strong', 'partial_downsample',                                   
                                   'grouped_cyclic_block', 'grouped_cyclic_angle_delay'])
    parser.add_argument('--RB_within_group', type = int, default = 4)
    parser.add_argument('--M_within_group', type = int, default = 1)
    
    return parser

