
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

%% Visualization

model_name = {'LSTM', 'Transformer', 'Linformer', ...
    'FEDformer', 'TimesNet', ...
    'PatchTST', 'iTransformer', ...
    'Dformer'};
model_num = length(model_name);

pre_nodes_ = {};
gt_nodes_ = {};

for iter_model = 1 : model_num
    path_kind = ['QuaDRiGa\', model_name{iter_model}, '\eval_result_92mats.mat'];
    load(path_kind);
    pre_nodes_{iter_model} = pre_nodes;
    gt_nodes_{iter_model} = gt_nodes;
end

% Rank channel elements by the ground-truth mean magnitude over the horizon.
gt_nodes = gt_nodes_{1};
batch_num = size(gt_nodes, 1);
pre_len = size(gt_nodes, 2);
RB_num = size(gt_nodes, 3);
antenna_num = size(gt_nodes, 4);
assert(numel(color_cell) >= model_num && numel(maker_cell) >= model_num, ...
    'Each model needs an entry in color_cell and maker_cell.');

while true
    batch_idx = input(sprintf('Enter batch_idx (1-%d, 0 to exit): ', batch_num));
    if isnumeric(batch_idx) && isreal(batch_idx) && isscalar(batch_idx) && batch_idx == 0
        break;
    end
    if ~isnumeric(batch_idx) || ~isreal(batch_idx) || ~isscalar(batch_idx) || ...
            ~isfinite(batch_idx) || batch_idx ~= fix(batch_idx) || ...
            batch_idx < 1 || batch_idx > batch_num
        fprintf('Please enter an integer between 0 and %d.\n', batch_num);
        continue;
    end

    % Edit these magnitude ranks to select the elements to visualize.
    vis_idx = {1};
    vis_ranks = cell2mat(vis_idx);
    assert(isnumeric(vis_ranks) && isreal(vis_ranks) && ...
        all(isfinite(vis_ranks(:))) && ...
        all(vis_ranks(:) == fix(vis_ranks(:))) && ...
        all(vis_ranks(:) >= 1 & vis_ranks(:) <= RB_num * antenna_num), ...
        'vis_idx must contain integer ranks from 1 to RB_num * antenna_num.');

    % Reshape explicitly to retain singleton horizon/RB/antenna dimensions.
    gt_complex = complex( ...
        reshape(gt_nodes(batch_idx, :, :, :, 1), [pre_len, RB_num, antenna_num]), ...
        reshape(gt_nodes(batch_idx, :, :, :, 2), [pre_len, RB_num, antenna_num]));
    mean_magnitude = reshape(mean(abs(gt_complex), 1), [RB_num, antenna_num]);
    [~, sorted_idx] = sort(mean_magnitude(:), 'descend');

    for iter_vis = 1 : numel(vis_ranks)
        magnitude_rank = vis_ranks(iter_vis);
        [RB_idx, antenna_idx] = ind2sub([RB_num, antenna_num], ...
            sorted_idx(magnitude_rank));

        for part_idx = 1 : 1
            if part_idx == 1
                part_name = 'Real';
            else
                part_name = 'Imaginary';
            end

            figure_num = figure_num + 1;
            figure(figure_num);
            clf;
            hold on;
            for iter_model = 1 : model_num
                trajectory = reshape(pre_nodes_{iter_model}( ...
                    batch_idx, :, RB_idx, antenna_idx, part_idx), 1, pre_len);
                plot(1 : pre_len, trajectory, maker_cell{model_num - iter_model + 1}, ...
                    'LineWidth', 1.5, 'Color', color_cell{model_num - iter_model + 1});
            end
            gt_trajectory = reshape(gt_nodes( ...
                batch_idx, :, RB_idx, antenna_idx, part_idx), 1, pre_len);
            plot(1 : pre_len, gt_trajectory, 'ko-', 'LineWidth', 2);
            hold off;

            xlabel('Prediction horizon', 'FontSize', 12, 'Interpreter', 'latex');
            ylabel([part_name, ' part of channel element'], 'FontSize', 12, 'Interpreter', 'latex');
            title(sprintf('Batch %d | Magnitude rank %d | RB %d | Antenna %d', ...
                batch_idx, magnitude_rank, RB_idx, antenna_idx));
            grid on;
            if pre_len > 1
                xlim([1, pre_len]);
            end
            xticks(1 : pre_len);
            legend([model_name, {'Ground truth'}], ...
                'FontSize', 10, 'Interpreter', 'latex', 'Location', 'best');
        end
    end
    drawnow;
end
