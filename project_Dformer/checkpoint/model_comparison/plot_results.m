
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

%% Fig.1 model comparison

dataset_name = {'QuaDRiGa', 'DeepMIMO', 'Industrial_Radio'};
dataset_num = length(dataset_name);
model_name = {'LSTM', 'Transformer', 'Linformer', ...
    'FEDformer', 'TimesNet', ...
    'PatchTST', 'iTransformer', ...
    'Dformer'};
model_num = length(model_name);

nmse_kind = zeros(model_num, dataset_num);

for iter_model = 1 : model_num
    for iter_dataset = 1 : dataset_num
        path_kind = [dataset_name{iter_dataset}, '\', model_name{iter_model}, '\eval_result_92mats.mat'];
        load(path_kind);
        nmse_kind(iter_model, iter_dataset) = mean(nmse_eval);
    end
end

figure_num = figure_num + 1;
figure(figure_num)
data = 10 * log10(nmse_kind);
hBar = bar(data); % 返回 bar 对象数组
for k = 1:length(hBar)
    hBar(k).FaceColor = color_cell{k};   % color_cell 是 cell，里面存 RGB 三元组
end
grid on;
xticks(1 : model_num)
xticklabels(model_name)
%xlabel('(SD density, FD density)', 'FontSize', 12, 'Interpreter','latex');
ylabel('NMSE (dB)', 'FontSize', 12, 'Interpreter','latex');
%ylim([0, 0.7])
ax = gca;
ax.XAxis.FontSize = 12;
ax.XAxis.FontName = 'Times New Roman';
ax.XAxis.TickLabelInterpreter = 'latex';
ax.YAxis.FontSize = 12;           % 设置字体大小
ax.YAxis.FontName = 'Times New Roman'; % 设置字体
ax.YAxis.TickLabelInterpreter = 'latex'; % 设置解释器为 LaTeX
legend({'QuaDRiGa', 'DeepMIMO', 'Industrial Radio'}, ...
        'FontSize', 10, 'Location', 'southwest', 'Interpreter', 'latex');

% 在每个柱顶端稍微往下打印数值
for k = 1:numel(hBar)
    xData = hBar(k).XData + hBar(k).XOffset; 
    yData = hBar(k).YData;
    for i = 1:length(yData)
        text(xData(i), yData(i) - 0.05, sprintf('%.3f', yData(i)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...   % 让文字贴在柱顶下方
            'FontSize', 12, ...
            'Interpreter', 'latex');
    end
end
