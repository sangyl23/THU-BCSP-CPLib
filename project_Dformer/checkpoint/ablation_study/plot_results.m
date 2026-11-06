
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

origin_path = pwd;

kind_name = {'lenc2', 'lenc4', 'dmodel64', 'dmodel192', ...
    'time', 'dopplersigma0p2', 'dopplersigma0p4', 'dopplersigma0p6', 'dopplersigma0p8', 'dopplersigma1', ...
    'ffnffn','attnattn', 'ffnattn', ...
    'variable_emb', ...
    'dopplerbeforeemb'};
kind_num = length(kind_name);

nmse_kind = zeros(kind_num, 1);

for iter_kind = 1 : kind_num
    path_kind = [kind_name{iter_kind}, '\eval_result_92mats.mat'];
    load(path_kind);
    nmse_kind(iter_kind) = mean(nmse_eval);
end

figure_num = figure_num + 1;
figure(figure_num)
data = 10 * log10(nmse_kind(:));   % 转成列向量
data = diag(data);                 % 转成对角矩阵，每列只有一个非零值
hBar = bar(data, 'stacked');       % 每列一个系列
grid on;
%xticks(1:2)
%xticklabels({'(SD density, FD density) = (1/2, 1/2)', '(SD density, FD density) = (1/4, 1/2)'})
ylabel('NMSE (dB)', 'FontSize', 13, 'Interpreter','latex');
%ylim([-13, 0])
ax = gca;
ax.XTick = [];       % 去掉刻度
ax.XTickLabel = [];  % 去掉刻度标签
ax.YAxis.FontSize = 12;           % 设置字体大小
ax.YAxis.FontName = 'Times New Roman'; % 设置字体
ax.YAxis.TickLabelInterpreter = 'latex'; % 设置解释器为 LaTeX
legend({'lenc2', 'lenc4', 'dmodel64', 'dmodel192', ...
    'time', 'dopplersigma0p2', 'dopplersigma0p4', 'dopplersigma0p6 (ours)', 'dopplersigma0p8', 'dopplersigma0p1', ...
    'ffnffn','attnattn', 'ffnattn', ...
    'variable emb', ...
    'dopplerbeforeemb'}, ...
    'FontSize', 10, 'Location', 'northwest', 'Interpreter', 'latex');
% 在每个柱顶端稍微往下打印数值
for k = 1:numel(hBar)
    xData = hBar(k).XData + hBar(k).XOffset; 
    yData = hBar(k).YData;
    idx = yData ~= 0;  % 找到非零位置
    text(xData(idx), yData(idx) - 0.05, string(sprintfc('%.3f', yData(idx))), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', ...
        'FontSize', 12, ...
        'Interpreter', 'latex');
end
