
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

%% NMSE

kind_name = {'full_long_period', 'partial_strong', 'grouped_cyclic_angle_delay'};
kind_num = length(kind_name);
SD_FD_density = {'SD1of2_FD1of1', 'SD1of2_FD1of2', 'SD1of4_FD1of2'};
density_num = length(SD_FD_density);

nmse_kind = zeros(kind_num, density_num);

for iter_kind = 1 : kind_num
    for iter_density = 1 : density_num
        path_kind = ['nmse\', kind_name{iter_kind}, '\', SD_FD_density{iter_density}, '\eval_result_92mats.mat'];
        load(path_kind);
        nmse_kind(iter_kind, iter_density) = mean(nmse_eval);
    end
end

nmse_kind = nmse_kind.';

path_kind = 'nmse\full\SD1of1_FD1of1\eval_result_92mats.mat';
load(path_kind);
full = mean(nmse_eval);

figure_num = figure_num + 1;
figure(figure_num)
hold on;

% 设置统一的柱宽，例如 0.8

hBar = bar(1, 10*log10(nmse_kind(3,:)), 'BarWidth', 0.8);
for k = 1:length(hBar)
    hBar(k).FaceColor = color_cell{kind_num - k + 1};   % color_cell 是 cell，里面存 RGB 三元组
end

hBar = bar(2, 10*log10(nmse_kind(2,:)), 'BarWidth', 0.8);
for k = 1:length(hBar)
    hBar(k).FaceColor = color_cell{kind_num - k + 1};   % color_cell 是 cell，里面存 RGB 三元组
end

hBar = bar(3, 10*log10(nmse_kind(1,:)), 'BarWidth', 0.8);
for k = 1:length(hBar)
    hBar(k).FaceColor = color_cell{kind_num - k + 1};   % color_cell 是 cell，里面存 RGB 三元组
end

% Match the grouped-bar layout so the visible bar has exactly the same width.
full_values = nan(1, numel(hBar));
middle_bar = ceil(numel(hBar) / 2);
full_values(middle_bar) = 10*log10(full);
full_bars = bar(4, full_values, 'FaceColor', color_gray, ...
    'BarWidth', hBar(1).BarWidth);
b1 = full_bars(middle_bar);

grid on;
xlim([0.5, 4.5])
xticks(1:4)
xticklabels({'$1/8$', '$1/4$', '$1/2$', '$1$'})
ax = gca;
ax.XAxis.FontSize = 12;
ax.XAxis.FontName = 'Times New Roman';
ax.XAxis.TickLabelInterpreter = 'latex';
ax.YAxis.FontSize = 12;
ax.YAxis.FontName = 'Times New Roman';
ax.YAxis.TickLabelInterpreter = 'latex';
xlabel('Pilot density $\gamma$', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('NMSE (dB)', 'FontSize', 14, 'Interpreter','latex');

lgd = legend([hBar, b1], ...
    {'Full port w/ long period [30]', 'Partial strongest port [20]', 'Grouped cyclic port (ours)', 'Full port [10]'}, ...
        'FontSize', 13, 'Interpreter', 'latex', 'Location', 'northeast');

%% SGCS

kind_name = {'full_long_period', 'partial_strong', 'grouped_cyclic_angle_delay'};
kind_num = length(kind_name);
SD_FD_density = {'SD1of2_FD1of1', 'SD1of2_FD1of2', 'SD1of4_FD1of2'};
density_num = length(SD_FD_density);

sgcs_kind = zeros(kind_num, density_num);

for iter_kind = 1 : kind_num
    for iter_density = 1 : density_num
        path_kind = ['sgcs\', kind_name{iter_kind}, '\', SD_FD_density{iter_density}, '\eval_result_92mats.mat'];
        load(path_kind);
        sgcs_kind(iter_kind, iter_density) = mean(sgcs_eval);
    end
end

sgcs_kind = sgcs_kind.';

path_kind = 'sgcs\full\SD1of1_FD1of1\eval_result_92mats.mat';
load(path_kind);
full = mean(sgcs_eval);

figure_num = figure_num + 1;
figure(figure_num)
hold on;

% 设置统一的柱宽，例如 0.8

hBar = bar(1, (sgcs_kind(3,:)), 'BarWidth', 0.8);
for k = 1:length(hBar)
    hBar(k).FaceColor = color_cell{kind_num - k + 1};   % color_cell 是 cell，里面存 RGB 三元组
end

hBar = bar(2, (sgcs_kind(2,:)), 'BarWidth', 0.8);
for k = 1:length(hBar)
    hBar(k).FaceColor = color_cell{kind_num - k + 1};   % color_cell 是 cell，里面存 RGB 三元组
end

hBar = bar(3, (sgcs_kind(1,:)), 'BarWidth', 0.8);
for k = 1:length(hBar)
    hBar(k).FaceColor = color_cell{kind_num - k + 1};   % color_cell 是 cell，里面存 RGB 三元组
end

% Match the grouped-bar layout so the visible bar has exactly the same width.
full_values = nan(1, numel(hBar));
middle_bar = ceil(numel(hBar) / 2);
full_values(middle_bar) = (full);
full_bars = bar(4, full_values, 'FaceColor', color_gray, ...
    'BarWidth', hBar(1).BarWidth);
b1 = full_bars(middle_bar);

grid on;
xlim([0.5, 4.5])
xticks(1:4)
xticklabels({'$1/8$', '$1/4$', '$1/2$', '$1$'})
ax = gca;
ax.XAxis.FontSize = 12;
ax.XAxis.FontName = 'Times New Roman';
ax.XAxis.TickLabelInterpreter = 'latex';
ax.YAxis.FontSize = 12;
ax.YAxis.FontName = 'Times New Roman';
ax.YAxis.TickLabelInterpreter = 'latex';
xlabel('Pilot density $\gamma$', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('SGCS', 'FontSize', 14, 'Interpreter','latex');

lgd = legend([hBar, b1], ...
    {'Full port w/ long period [30]', 'Partial strongest port [20]', 'Grouped cyclic port (ours)', 'Full port [10]'}, ...
        'FontSize', 13, 'Interpreter', 'latex', 'Location', 'southwest');

%% SE

kind_name = {'full_long_period', 'partial_strong', 'grouped_cyclic_angle_delay'};
kind_num = length(kind_name);
SD_FD_density = {'SD1of2_FD1of1', 'SD1of2_FD1of2', 'SD1of4_FD1of2'};
density_num = length(SD_FD_density);

se_kind = zeros(kind_num, density_num);

for iter_kind = 1 : kind_num
    for iter_density = 1 : density_num
        path_kind = ['se\', kind_name{iter_kind}, '\', SD_FD_density{iter_density}, '\eval_result_92mats.mat'];
        load(path_kind);
        se_kind(iter_kind, iter_density) = mean(se_eval);
    end
end

se_kind = se_kind.';

path_kind = 'se\full\SD1of1_FD1of1\eval_result_92mats.mat';
load(path_kind);
full = mean(se_eval);

figure_num = figure_num + 1;
figure(figure_num)
hold on;

% 设置统一的柱宽，例如 0.8

hBar = bar(1, (se_kind(3,:)), 'BarWidth', 0.8);
for k = 1:length(hBar)
    hBar(k).FaceColor = color_cell{kind_num - k + 1};   % color_cell 是 cell，里面存 RGB 三元组
end

hBar = bar(2, (se_kind(2,:)), 'BarWidth', 0.8);
for k = 1:length(hBar)
    hBar(k).FaceColor = color_cell{kind_num - k + 1};   % color_cell 是 cell，里面存 RGB 三元组
end

hBar = bar(3, (se_kind(1,:)), 'BarWidth', 0.8);
for k = 1:length(hBar)
    hBar(k).FaceColor = color_cell{kind_num - k + 1};   % color_cell 是 cell，里面存 RGB 三元组
end

% Match the grouped-bar layout so the visible bar has exactly the same width.
full_values = nan(1, numel(hBar));
middle_bar = ceil(numel(hBar) / 2);
full_values(middle_bar) = (full);
full_bars = bar(4, full_values, 'FaceColor', color_gray, ...
    'BarWidth', hBar(1).BarWidth);
b1 = full_bars(middle_bar);

grid on;
xlim([0.5, 4.5])
xticks(1:4)
xticklabels({'$1/8$', '$1/4$', '$1/2$', '$1$'})
ax = gca;
ax.XAxis.FontSize = 12;
ax.XAxis.FontName = 'Times New Roman';
ax.XAxis.TickLabelInterpreter = 'latex';
ax.YAxis.FontSize = 12;
ax.YAxis.FontName = 'Times New Roman';
ax.YAxis.TickLabelInterpreter = 'latex';
xlabel('Pilot density $\gamma$', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('SE (bps/Hz)', 'FontSize', 14, 'Interpreter','latex');

lgd = legend([hBar, b1], ...
    {'Full port w/ long period [30]', 'Partial strongest port [20]', 'Grouped cyclic port (ours)', 'Full port [10]'}, ...
        'FontSize', 13, 'Interpreter', 'latex', 'Location', 'southwest');