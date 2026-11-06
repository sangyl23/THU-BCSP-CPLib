# Doppler-Domain Transformer-Based Channel Prediction with Grouped Cyclic Port Estimation

**Hugging Face downloads:** [Test dataset](https://huggingface.co/datasets/sangyl23/THU_BCSP_CPLib/tree/main/project_Dformer/QuaDRiGa) | [Checkpoints and evaluation commands](https://huggingface.co/datasets/sangyl23/THU_BCSP_CPLib/tree/main/project_Dformer/checkpoint) | [Full artifact repository](https://huggingface.co/datasets/sangyl23/THU_BCSP_CPLib)

PyTorch implementation accompanying **?Doppler-Domain Transformer-Based Channel Prediction with Grouped Cyclic Port Estimation.?** This release provides evaluation code, a QuaDRiGa test dataset, and pretrained checkpoints. Large artifacts are hosted on Hugging Face and downloaded separately from the GitHub source code.

> **Release schedule:** The training datasets, training-data generation code, and Python training scripts will be released immediately after the paper receives a major-revision decision. This README describes evaluation with pretrained checkpoints only; no training dataset is required for the example below.

## Model overview

[View the model architecture (PDF)](./fig2_model_illustration.pdf)

The figure contrasts variable embedding with Dformer's temporal embedding and illustrates the Dopplercut, variable-attention / temporal-FFN encoder, and Dopplerfill modules. Dopplercut removes high-Doppler components; Dopplerfill restores the sequence length through zero filling before conversion back to the time domain.

## 1. Clone the source code and configure Conda

Run these commands in Anaconda Prompt on Windows:

```bat
git clone https://github.com/sangyl23/THU_BCSP_CPLib.git
cd THU_BCSP_CPLib
conda create -n cplib python=3.11.14 pip=25.3 -y
conda activate cplib
python -m pip install -r requirements.txt
python -m pip check
```

The pinned dependency file snapshots a Windows x86-64 / Python 3.11.14 environment with PyTorch 2.5.1 (CPU). The example below explicitly uses CPU and may take substantially longer than GPU evaluation. For CUDA evaluation, install the corresponding official CUDA-enabled PyTorch 2.5.1 build for your hardware and change `--gpu cpu` to `--gpu cuda:0`. A CUDA installation changes the CPU environment snapshot. Other platforms have not been validated with this dependency file.

## 2. Download the test dataset and checkpoints

Run the following commands **from the repository root**, in the activated `cplib` environment. They use `huggingface_hub`, which is already included in `requirements.txt`; public downloads do not require login.

Download the released test split:

```bat
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sangyl23/THU_BCSP_CPLib', repo_type='dataset', allow_patterns=['project_Dformer/QuaDRiGa/process_v40kmh_1en3_his64_pre8/test_ULsnr10dB_DLsnr20dB/**'], local_dir='.')"
```

Download the checkpoint directory, including `.pth` weights, configurations recorded in command files, and existing logs:

```bat
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sangyl23/THU_BCSP_CPLib', repo_type='dataset', allow_patterns=['project_Dformer/checkpoint/**'], local_dir='.')"
```

`local_dir='.'` preserves the paths stored on Hugging Face. Do not download from inside `project_Dformer`, as this would create an extra nested `project_Dformer` directory. The resulting layout should be:

```text
THU_BCSP_CPLib/
??? README.md
??? fig2_model_illustration.pdf
??? requirements.txt
??? models/
??? layers/
??? utils/
??? project_Dformer/
    ??? configs/
    ??? dataloader_cplib.py
    ??? eval_cplib.py
    ??? QuaDRiGa/
    ?   ??? process_v40kmh_1en3_his64_pre8/
    ?       ??? test_ULsnr10dB_DLsnr20dB/
    ?           ??? dataset_*.mat
    ??? checkpoint/
        ??? model_comparison/
        ?   ??? QuaDRiGa/
        ?       ??? Dformer/
        ?           ??? Dformer_QuaDRiGa_v40kmh_interval1en392mats_snr20dB_tt0.pth
        ?           ??? running_eval.txt
        ??? port_estimation_mode/
        ??? ablation_study/
        ??? velocity/
        ??? snr/
```

The released split corresponds to 40 km/h, a 1 ms sampling interval, uplink SNR 10 dB, and downlink SNR 20 dB. The examples below use this split. Commands for other velocities, SNRs, or channel datasets require their corresponding test data; downloading their checkpoints alone is insufficient.

## 3. Evaluate Dformer

Switch to the evaluation working directory:

```bat
cd project_Dformer
```

The following command is adapted from `checkpoint/model_comparison/QuaDRiGa/Dformer/running_eval.txt`, with the device changed to CPU. **Copy it as a single line** in Anaconda Prompt or PowerShell:

```text
python eval_cplib.py --model_name Dformer --channel_type QuaDRiGa --v 40kmh --interval 1en3 --snr 20dB --UE_num 4 --RB_num 8 --M 32 --M_h 8 --M_v 2 --seq_len 64 --pred_len 8 --label_len 32 --t0 64 --read_uplink False --path_train QuaDRiGa/process_v40kmh_1en3_his64_pre8/train_ULsnr10dB_DLsnr20dB --path_eval QuaDRiGa/process_v40kmh_1en3_his64_pre8/test_ULsnr10dB_DLsnr20dB --port_estimation_mode grouped_cyclic_angle_delay --RB_within_group 4 --M_within_group 16 --training_data 92mats --epoch_num 300 --batch_size 16 --gpu cpu --multi_gpu False --cuda_devices 0,1,2,3 --loss_function NMSE --lr 5e-4 --seed 1 --evaluate_sgcs True --evaluate_se True --saved_path checkpoint/model_comparison/QuaDRiGa/Dformer/ --input_mechanism Channel_mixing --instance_norm wo --ifAD True --task_name long_term_forecast --enc_in 128 --dec_in 128 --c_out 128 --e_layers 3 --d_layers 0 --d_model 128 --d_ff 64 --ifemb_conditions False --emb_module temporal_emb --embed_learning doppler --doppler_keep_ratio 0.6 --checkpoint_name Dformer_QuaDRiGa_v40kmh_interval1en392mats_snr20dB_tt0
```

This loads the pretrained Dformer checkpoint and evaluates grouped cyclic angle-delay port estimation with `RB_within_group=4` and `M_within_group=16` (four groups for eight RBs and 32 antenna ports). NMSE, SGCS, and SE evaluation are enabled.

Important path and argument conventions:

- Run from `project_Dformer`, so `QuaDRiGa/...` and `checkpoint/...` resolve correctly.
- `--saved_path` is both the checkpoint directory and output directory. Keep its trailing `/` because the script concatenates file names directly.
- `--checkpoint_name` is the checkpoint file name **without** `.pth`.
- The inherited `--path_train`, `--training_data`, `--epoch_num`, and learning-rate arguments are retained from the recorded command. They do not trigger training; `eval_cplib.py` constructs only the evaluation loader. In particular, `92mats` is a training-run label, not a requirement to download a training set.
- The original `running_eval.txt` files use Bash-style `\` line continuation. On Windows, use the single-line command above, or join the lines before executing. Do not paste the multiline Bash form directly into PowerShell or Anaconda Prompt.

## 4. Inspect results and run another evaluation

The example writes:

```text
project_Dformer/checkpoint/model_comparison/QuaDRiGa/Dformer/eval_result_92mats.mat
```

The MATLAB file contains `nmse_eval`, `sgcs_eval`, `se_eval`, `pre_nodes`, and `gt_nodes`. The script also writes `logfile.txt` in the selected `--saved_path`. Re-running an evaluation overwrites these output files. Large `eval_result_92mats.mat` files are excluded from the published artifacts and regenerated locally during evaluation.

For the grouped cyclic port-estimation comparison at the same group sizes, use the identical command from Section 3 but replace:

```text
--saved_path checkpoint/model_comparison/QuaDRiGa/Dformer/
```

with:

```text
--saved_path checkpoint/port_estimation_mode/nmse/grouped_cyclic_angle_delay/SD1of2_FD1of2/
```

This matches the corresponding `running_eval.txt`; the checkpoint name and remaining arguments are the same for this example.

For other configurations, use the `running_eval.txt` next to the desired checkpoint as the source of model, grouping, metric, and data-path arguments. Keep the architecture parameters consistent with the selected `.pth` file, and confirm that its test data are available before running. A checkpoint directory may also contain historical `running.txt` training commands; these are not part of the evaluation workflow in this release.
