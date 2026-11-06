
clear all, close all, clc;

%
color_b = [0 0.4470 0.7410];
color_r = [1 0 0];
color_y = [0.9290 0.6940 0.1250];
color_p = [0.4940 0.1840 0.5560];
color_g = [0.4660 0.6740 0.1880];    
color_deepred = [0.6350 0.0780 0.1840];
color_orange = [0.8500 0.3250 0.0980];
color_lightblue = [0.3010 0.7450 0.9330];    % 浅蓝色
color_magenta = [0.6350 0.0780 0.9800];      % 品红色
color_lime = [0.8500 0.9000 0.0980];         % 淡绿色
color_teal = [0.2500 0.4500 0.4500];         % 蓝绿色
color_gold = [0.8540 0.6470 0.1250];         % 金黄色
color_gray = [0.5 0.5 0.5];                  % 中灰色
color_olive = [0.5960 0.7740 0.1840];        % 橄榄绿色
color_maroon = [0.5560 0.1840 0.2940];       % 栗色

color_cell = {color_r, color_b, color_p, color_orange, color_g, color_gray, color_gold, color_lightblue};
maker_cell = {'^-', '*-', '+-', 'd-', 's-', 'p-', 'h-', '>-'};

figure_num = 0;
epoch_num = 300;
pre_len = 8;

%% Figure snr

model_name = {'LSTM', 'Transformer', 'Linformer', 'FEDformer', ...
    'TimesNet', 'PatchTST', 'iTransformer', 'Dformer'};
model_num = length(model_name);
snr_name = {'15dB', '20dB', ...
    '25dB', '30dB'};
snr_num = length(snr_name);

nmse_kind = zeros(model_num, snr_num);

for iter_model = 1 : model_num
    for iter_snr = 1 : snr_num
        path_kind = [model_name{iter_model}, '\', snr_name{iter_snr}, '\eval_result_92mats.mat'];
        load(path_kind);
        nmse_kind(iter_model, iter_snr) = mean(nmse_eval);
    end
end

figure_num = figure_num + 1;
figure(figure_num)
hold on;
for iter_model = 1 : model_num
    plot(1 : 1 : snr_num, 10 * log10(nmse_kind(iter_model, :)), maker_cell{model_num - iter_model + 1}, 'linewidth', 1.5, 'Color', color_cell{model_num - iter_model + 1});
end
grid on;
xlabel('Downlink SNR (dB)', 'FontSize', 12, 'interpreter',' latex');
ylabel('NMSE (dB)', 'FontSize', 12, 'interpreter',' latex');
xlim([1, snr_num])
xticks(1 : 1 : snr_num)
xticklabels({'15', '20', '25', '30'})
%ylim([-6.5, 0])
legend({'LSTM [8]', 'Transformer [9]', ...
        'LinFormer [13]', 'FEDformer [25]', ...
        'TimesNet [26]', 'PatchTST [27]', ...
        'iTransformer [28]', 'Dformer (ours)'}, ...
        'FontSize', 10, 'Interpreter', 'latex', 'Location', 'northeast');
% 调整 xticklabels 的字体和解释器
ax = gca; % 获取当前坐标轴
ax.XAxis.FontSize = 12;           % 设置字体大小
ax.XAxis.FontName = 'Times New Roman'; % 设置字体
ax.XAxis.TickLabelInterpreter = 'latex'; % 设置解释器为 LaTeX
ax.YAxis.FontSize = 12;           % 设置字体大小
ax.YAxis.FontName = 'Times New Roman'; % 设置字体
ax.YAxis.TickLabelInterpreter = 'latex'; % 设置解释器为 LaTeX