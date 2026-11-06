
def add_training_args(parser):
    # training parameters
    parser.add_argument('--training_data', type = str, default = '92mats')
    parser.add_argument('--training_time', type = int, default = 1)
    parser.add_argument('--epoch_num', type = int, default = 300)
    parser.add_argument('--eval_every_epochs', type=int, default=1,
                        help='Evaluate every N epochs; the final epoch is always evaluated')
    parser.add_argument('--batch_size', type = int, default = 2)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--gpu', type = str, default = 'cuda:0',
                        help = 'Primary CUDA device used for single-GPU training.')
    parser.add_argument('--multi_gpu', type = eval, default = False,
                        help = 'Enable torch.nn.DataParallel training.')
    parser.add_argument('--cuda_devices', type = str, default = '0,1',
                        help = 'Comma-separated CUDA device IDs, for example: 0,1,3.')
    parser.add_argument('--loss_function', type = str, default = 'NMSE',
                        choices = ['MSE', 'NMSE', 'NSGCS', 'NSE'])
    parser.add_argument('--lr', type = float, default = 5e-4)
    parser.add_argument('--seed', type = int, default = 0)
    
    parser.add_argument('--evaluate_sgcs', type = eval, default = False)
    parser.add_argument('--evaluate_se', type = eval, default = False)
   
    parser.add_argument('--saved_path', type = str, default = 'tmp/')
    parser.add_argument('--ifvis', type = eval, default = False,
                        help = 'Save six Doppler stages as .mat files and block until all plot windows close')
    # Control whether each batch returns speed and sampling-interval conditions.
    parser.add_argument('--return_conditions', type = eval, default = True)
       
    return parser
