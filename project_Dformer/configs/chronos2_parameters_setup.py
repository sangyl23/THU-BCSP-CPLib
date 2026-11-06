
def add_model_args(parser):
    # model parameters
    parser.add_argument('--dropout', type = float, default = 0.08)
    parser.add_argument('--input_mechanism', type = str, default = 'Channel_mixing',
                        choices = ['Channel_independent', 'Channel_mixing', 'RB_independent', 'M_independent'])
    parser.add_argument('--instance_norm', type = str, default = 'RevIN',
                        choices = ['norm_2', 'norm_max', 'RevIN', 'wo'])
    parser.add_argument('--ifAD', type = eval, default = True)
    
    # Transformer parameters
    parser.add_argument('--task_name', type = str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection, zero_shot_forecast]')
    parser.add_argument('--quantile_levels', type=float, nargs='+', default=[0.1, 0.5, 0.9],
                        help='Chronos-2 forecast quantiles, for example: --quantile_levels 0.1 0.5 0.9')
    parser.add_argument('--chronos_model_path', type=str, default='pretrained/chronos-2',
                        help='Project-relative or absolute path to the local Chronos-2 snapshot')
    parser.add_argument('--chronos_device', type=str, default='cuda',
                        help='Device map passed to Chronos-2, for example cuda, cuda:0, or cpu')
    parser.add_argument('--chronos_dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--chronos_batch_size', type=int, default=256,
                        help='Maximum number of channel variates processed per Chronos-2 inference batch')
    parser.add_argument('--chronos_layers', type=int, default=12,
                        help='Number of pretrained Chronos-2 encoder layers to keep, from the first layer onward')

    # Chronos-2 selective fine-tuning. All switches default to False so that
    # zero-shot inference behaves exactly like the original pretrained model.
    parser.add_argument('--iffinetune_all', type=eval, default=False,
                        help='Fine-tune every Chronos-2 parameter; overrides partial selections')
    parser.add_argument('--iffinetune_ln', type=eval, default=False,
                        help='Fine-tune LayerNorm scales in selected encoder layers and final LayerNorm')
    parser.add_argument('--iffinetune_attention', type=eval, default=False,
                        help='Fine-tune temporal and cross-variable group-attention projections')
    parser.add_argument('--iffinetune_mlp', type=eval, default=False,
                        help='Fine-tune Transformer feed-forward layers')
    parser.add_argument('--iffinetune_input_embedding', type=eval, default=False,
                        help='Fine-tune input patch and registration-token embeddings')
    parser.add_argument('--iffinetune_output_embedding', type=eval, default=False,
                        help='Fine-tune the output quantile prediction head')
    parser.add_argument('--chronos_trainable_layers', type=int, default=-1,
                        help='Apply LN/attention/MLP tuning to the last N retained layers; -1 selects all, 0 selects none')
    parser.add_argument('--chronos_lr', type=float, default=1e-5,
                        help='Learning rate for trainable Chronos-2 parameters')
    parser.add_argument('--ifchronos_lr_warmup_clipping', type=eval, default=True,
                        help='Enable learning-rate warmup and gradient clipping for Chronos-2')
    parser.add_argument('--chronos_warmup_ratio', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    
    return parser
