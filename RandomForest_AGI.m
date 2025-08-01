%*************************************************************************%
%*************************Feature Extraction******************************%
%*************************************************************************%
clc
clear;
%% 1. Preparation for dataset
%first，准备你的数据集，将原始电流数据存储进去
path_original_arc = 'E:\资料\AFCI_Train_dataset\Original_data\Arc\';  
%arc fault数据文件夹地址
path_original_norm = 'E:\资料\AFCI_Train_dataset\Original_data\Normal\';  
%normal数据文件夹地址

%% 2.Obtain the information of all the .csv files under the current 
% data path
%Traverse all the files in the Arc folder and save the current data 
% contained in the files.
[arc_samples] = scanDir(path_original_arc);
arc_samples_num = length(arc_samples);

%Traverse all the files in the Normal folder and save the current 
% data contained in the files.
[norm_samples] = scanDir(path_original_norm);
norm_samples_num = length(norm_samples);

%% 3. Feature extraction
%主程序 - 特征提取
% 定义矩阵保存time domain features
arc_pp = zeros(arc_samples_num, 4);
arc_rms = zeros(arc_samples_num, 4);
arc_std = zeros(arc_samples_num, 4);
arc_cf = zeros(arc_samples_num, 4);
arc_ae = zeros(arc_samples_num, 4);

norm_pp = zeros(norm_samples_num, 4);
norm_rms = zeros(norm_samples_num, 4);
norm_std = zeros(norm_samples_num, 4);
norm_cf = zeros(norm_samples_num, 4);
norm_ae = zeros(norm_samples_num, 4);

% 定义元胞数组保存频谱（每行一个样本，每列一个差分阶数）
arc_spectra = cell(arc_samples_num, 4);
norm_spectra = cell(norm_samples_num, 4);

% 处理arc样本
for k = 1:arc_samples_num
    % 读取数据
    data = readmatrix(arc_samples{k});

    % 计算特征
    [pp_vals, rms_vals, std_vals, cf_vals, ae_vals, spec] = ...
        calculateFeatures(data, 1, 0.15);

    % 存储时域特征
    arc_pp(k, :) = pp_vals;
    arc_rms(k, :) = rms_vals;
    arc_std(k, :) = std_vals;
    arc_cf(k, :) = cf_vals;
    arc_ae(k, :) = ae_vals;

    % 存储频域特征
    arc_spectra(k, :) = spec;
end

% 处理norm样本
for k = 1:norm_samples_num
    % 读取数据
    data = readmatrix(norm_samples{k});

    % 计算特征
    [pp_vals, rms_vals, std_vals, cf_vals, ae_vals, spec] = ...
        calculateFeatures(data, 1, 0.15);

    % 存储时域特征
    norm_pp(k, :) = pp_vals;
    norm_rms(k, :) = rms_vals;
    norm_std(k, :) = std_vals;
    norm_cf(k, :) = cf_vals;
    norm_ae(k, :) = ae_vals;

    % 存储频域特征
    norm_spectra(k, :) = spec;
end
%% 4. 组合特征向量
% 计算每个样本的特征向量总长度
time_features_per_sample = 4 * 5;  % 5种时域特征 × 4个差分阶数
spec_lengths = cellfun(@length, arc_spectra(1, :));  % 获取第一个样本的频谱长度
spec_length_total = sum(spec_lengths);  % 每个样本的频谱总长度
feature_vec_length = time_features_per_sample + spec_length_total;  % 每个样本的特征向量总长度

% 初始化特征矩阵（优化内存分配）
arc_feature = zeros(arc_samples_num, feature_vec_length);
norm_feature = zeros(norm_samples_num, feature_vec_length);

% 组合特征向量
for k = 1:arc_samples_num
    % 1. 组合时域特征
    time_features = [arc_pp(k, :), ...   % 峰峰值 (4个)
                   arc_rms(k, :), ...   % RMS值 (4个)
                   arc_std(k, :), ...   % 标准差 (4个)
                   arc_cf(k, :), ...    % 峰值因子 (4个)
                   arc_ae(k, :)];       % 近似熵 (4个)

    % 2. 组合频域特征（频谱）
    spec_features = [arc_spectra{k, 1}; ...  % 原始信号频谱
                    arc_spectra{k, 2}; ...  % 一阶差分频谱
                    arc_spectra{k, 3}; ...  % 二阶差分频谱
                    arc_spectra{k, 4}];     % 三阶差分频谱

    % 3. 合并时域和频域特征
    arc_feature(k, :) = [time_features, spec_features'];
end

for k = 1:norm_samples_num
    % 1. 组合时域特征
    time_features = [norm_pp(k, :), ...   % 峰峰值 (4个)
                   norm_rms(k, :), ...   % RMS值 (4个)
                   norm_std(k, :), ...   % 标准差 (4个)
                   norm_cf(k, :), ...    % 峰值因子 (4个)
                   norm_ae(k, :)];       % 近似熵 (4个)

    % 2. 组合频域特征（频谱）
    spec_features = [norm_spectra{k, 1}; ...  % 原始信号频谱
                    norm_spectra{k, 2}; ...  % 一阶差分频谱
                    norm_spectra{k, 3}; ...  % 二阶差分频谱
                    norm_spectra{k, 4}];     % 三阶差分频谱

    % 3. 合并时域和频域特征
    norm_feature(k, :) = [time_features, spec_features'];
end

%% 5.定义聚类参数

%定义存储样本的矩阵
dim = arc_samples_num + norm_samples_num;  % 模式样本维数
sample_t = zeros(dim,feature_vec_length);

for k=1:arc_samples_num
    sample_t(k,:) = arc_feature(k, :);
end

for k=1:norm_samples_num
    sample_t(k+arc_samples_num,:) = norm_feature(k, :);
end

% 设置聚类数
cluster_num = 4;
data = sample_t';

% 调用K-Means算法进行聚类
[m, n] = size(data); % 获取数据的行数和列数
cluster = data(randperm(m, cluster_num), :); % 从m个点中随机选择cluster_num个点作为初始聚类中心点
epoch_max = 1000; % 最大迭代次数
therad_lim = 0.001; % 中心变化阈值
epoch_num = 0; % 迭代次数初始化
E = inf; % 初始化准则函数E

while (epoch_num < epoch_max)
    epoch_num = epoch_num + 1; % 迭代次数加1

    %PCA降维
    data_init = zscore(data);   %对数据sample标准化为data_init，matlab内置的标准化函数（x-mean(x)）/std(x)
        
    % distance1存储每个点到各聚类中心的余弦距离
    for i = 1:cluster_num
        %将每个聚类中心数据复制m份，得到m行中心数据，再将所有的数据和该中心做差求平方
        distance = data .* repmat(cluster(i,:), m, 1);
        distance1 = data .^2;
        centerd = sum(cluster(i,:) .^2);
        distance2 = sqrt(sum(distance1, 2) .* centerd);
        distance3(:, i) = 1 - sum(distance, 2) ./ distance2; %将所有点到聚类中心的余弦距离存成一列
    end
    [~, index_cluster] = min(distance3, [], 2); % 找到每个点距离最近的聚类中心，min(A,[],2) 是包含每一行的最小值的列向量。
        
    % cluster_new存储新的聚类中心
    for j = 1:cluster_num
        cluster_new(j, :) = mean(data(find(index_cluster == j), :)); % 计算新的聚类中心
    end
        
    % 计算当前聚类结果的准则函数E
    E_new = 0;
    for j = 1:cluster_num
        E_new = E_new + sum(sum((data(find(index_cluster == j), :) - cluster(j, :)).^2));
    end
        
    % 如果新的聚类中心和上一轮的聚类中心距离和大于阈值，更新聚类中心，否则算法结束
    if (sqrt(sum((cluster_new - cluster).^2, 'all')) > therad_lim)
        cluster = cluster_new; % 更新聚类中心
        E = E_new; % 更新准则函数E
    else
        break; % 跳出循环，算法结束
    end
end
%% 6. 计算每个特征的AGI值并保存，计算每个特征中AGI值最小的10个特征所对应的index，并保存下来
%定义标签
Y_label = zeros(dim, 1);
for k = 1:arc_samples_num
    Y_label(k,1) = 1;
end

for k = 1:norm_samples_num
    Y_label(arc_samples_num + k,1) = 0;
end
% 获取每个特征的AGI值
feature_gini = Find_optimal_feature(sample_t, Y_label);

% 初始化存储每个类别中AGI最小的15个特征索引的矩阵
top20_index = cell(cluster_num, 1);  % 使用元胞数组存储不同长度的索引
top20_gini = cell(cluster_num, 1);   % 存储对应的AGI值

% 遍历所有特征
for k = 1:feature_vec_length
    % 获取当前特征所属的类别
    class_idx = index_cluster(k);
    
    % 初始化或更新该类的特征索引和AGI值列表
    if isempty(top20_index{class_idx})
        top20_index{class_idx} = k;
        top20_gini{class_idx} = feature_gini(k);
    else
        % 将当前特征添加到该类列表中
        top20_index{class_idx} = [top20_index{class_idx}, k];
        top20_gini{class_idx} = [top20_gini{class_idx}, feature_gini(k)];
    end
end

% 对每个类别的特征按AGI值排序并取前20个
for j = 1:cluster_num
    if ~isempty(top20_gini{j})
        [sorted_gini, sort_idx] = sort(top20_gini{j}); % 升序排序
        % 取AGI值最小的前15个特征
        if length(sorted_gini) > 20
            top20_index{j} = top20_index{j}(sort_idx(1:20));
        else
            top20_index{j} = top20_index{j}(sort_idx);
        end
    end
end

%% 7. 将筛选后特征组成新的样本
% 计算新特征的总维度
new_feature_dim = 0;
for j = 1:cluster_num
    new_feature_dim = new_feature_dim + length(top20_index{j});
end

% 定义新的样本矩阵
X_data = zeros(dim, new_feature_dim);

% 填充新样本矩阵
for i = 1:dim
    col_idx = 1;
    for j = 1:cluster_num
        if ~isempty(top20_index{j})
            % 取出当前类别选中的特征
            selected_features = sample_t(i, top20_index{j});
            feature_count = length(selected_features);
            % 存入新样本矩阵
            X_data(i, col_idx:col_idx+feature_count-1) = selected_features;
            col_idx = col_idx + feature_count;
        end
    end
end
%% 8.生成训练集和测试集
train_ratio = 0.7;
arc_train_num = fix(train_ratio*arc_samples_num);
arc_test_num = arc_samples_num - arc_train_num;

norm_train_num = fix(train_ratio*norm_samples_num);
norm_test_num = norm_samples_num - norm_train_num;

train_length = arc_train_num + norm_train_num;
test_length = arc_test_num + norm_test_num;

% 随机产生训练集和测试集
n = randperm(size(X_data,1));
% 训练集——arc_length_train个样本
train_matrix = X_data(n(1:train_length),:);
train_label = Y_label(n(1:train_length),:);
% 测试集——test_length个样本
test_matrix = X_data(n((train_length+1):end),:);
test_label = Y_label(n((train_length+1):end),:);

%% 9. 训练随机森林
%使用build_tree函数来训练决策树。
% 步骤：
% 如果当前节点是叶节点（包含 value 字段），返回叶节点的值。
% 否则，根据当前节点的特征和阈值，决定进入左子树还是右子树，递归调用 predict 函数。
min_samples_split = 5;       %定义叶子节点的最小样本数量
max_depth = 10;              %定义树的最大深度
numTrees = 15;     
numFeatures = 10;
RF = random_forest_train(train_matrix, train_label, numTrees, min_samples_split, max_depth, numFeatures);

%对测试集中的每个样本使用决策树tree预测输出，
predictions = arrayfun(@(i) random_forest_predict(RF, test_matrix(i, :)), 1:size(test_matrix, 1));
predictions_label = predictions';

%% 10.相关指标计算
R2 = 1 - norm(test_label - predictions_label)^2 / norm(test_label - mean(test_label))^2;
disp(['测试集数据的R2为：', num2str(R2)])

mae2 = sum(abs(predictions_label - test_label)) ./ test_length ;
disp(['测试集数据的MAE为：', num2str(mae2)])

mbe2 = sum(predictions_label - test_label ) ./ test_length ;
disp(['测试集数据的MBE为：', num2str(mbe2)])

%准确率
acc = 100*sum(predictions_label == test_label)./numel(test_label); 
% 错报故障率
err_report1 = 100*sum(predictions_label > test_label)./test_length;
% 漏报故障率
err_report2 = 100*sum(predictions_label < test_label)./test_length;

disp(['Accuracy rate of fault prediction is：', num2str(acc)])
disp(['Fault false alarm rate is：', num2str(err_report1)])
disp(['Fault underreporting rate is：', num2str(err_report2)])
%*************************************************************************%
%*************************Functions definition******************************%
%*************************************************************************%
% 1.定义通用函数计算特征
function [pp, rms_val, std_val, cf, ae, spectra] = calculateFeatures(data, m, r)
    % 输入: 
    %   data - 原始信号数据
    %   m - 近似熵嵌入维度
    %   r - 近似熵半径
    % 输出:
    %   时域特征和频域特征
    
    % 初始化特征向量
    pp = zeros(1,4);
    rms_val = zeros(1,4);
    std_val = zeros(1,4);
    cf = zeros(1,4);
    ae = zeros(1,4);
    spectra = cell(1,4);  % 存储每个差分信号的频谱
    
    % 计算多步差分信号
    diff_signals = cell(1,4);
    diff_signals{1} = data(:);          % 原始信号 (确保列向量)
    diff_signals{2} = diff(diff_signals{1});
    diff_signals{3} = diff(diff_signals{1}, 2);
    diff_signals{4} = diff(diff_signals{1}, 3);
    
    % 计算每个差分信号的特征
    for j = 1:4
        signal = diff_signals{j};
        
        % 时域特征
        pp(j) = max(signal) - min(signal);
        rms_val(j) = rms(signal);
        std_val(j) = std(signal, 1);
        cf(j) = max(abs(signal)) / rms_val(j);  % 使用绝对值计算峰值因子
        
        % 近似熵计算（带异常处理）
        try
            if length(signal) > 10  % 确保足够长度
                ae(j) = approximateEntropy(signal, 'Dimension', m, 'Radius', r);
            else
                ae(j) = NaN;
            end
        catch
            ae(j) = NaN;
        end
        
        % 频域特征（计算单边功率谱）
        spectra{j} = computePowerSpectrum(signal);
    end
end

% 2.辅助函数：计算单边功率谱
function spectrum = computePowerSpectrum(signal)
    L = length(signal);
    
    % 应用汉宁窗
    win = hanning(L);
    xw = signal .* win;
    
    % 计算FFT
    X = fft(xw, L);
    
    % 计算单边功率谱（正确归一化）
    P2 = abs(X/L);           % 双边幅度谱
    P1 = P2(1:floor(L/2)+1); % 取正频率部分
    P1(2:end-1) = 2*P1(2:end-1); % 调整幅度（除DC和Nyquist）
    
    % 计算功率谱密度 (dB/Hz)
    spectrum = 10*log10(P1.^2/(sum(win.^2)*L)); % 正确的功率谱归一化
end
% 3. 定义求取数据集中每个特征的gini系数函数
% X：特征矩阵
% Y：标签向量
% min_samples_split：节点分裂的最小样本数
% max_depth：树的最大深度
% depth：当前树的深度
function [ feature_gini ] = Find_optimal_feature(X, Y)
    [n_samples, n_features] = size(X);
    
    % 目标：遍历所有特征和可能的分割点，找到使基尼指数最小的分割。
    feature_gini = zeros(n_features,1);

    %步骤：
    %遍历每个特征（feature）。
    %对每个特征，遍历其所有唯一值作为候选分割点（threshold）。
    %根据当前特征和分割点，将数据集分为左右两部分：
    %left_indices：满足 X(:, feature) <= threshold 的样本。
    % right_indices：满足 X(:, feature) > threshold 的样本
    % 如果某一部分没有样本，跳过该分割点。
    % 计算分割后的基尼指数（gini_impurity），如果比当前最佳基尼指数小，则更新最佳分割。
    
    for feature = 1:n_features
        best_gini = inf;
        best_feature = 0;
        best_threshold = 0;
        thresholds = unique(X(:, feature));
        for threshold = thresholds'  %遍历所有不同的特征值
            left_indices = X(:, feature) <= threshold;
            right_indices = X(:, feature) > threshold;
            
            if sum(left_indices) == 0 || sum(right_indices) == 0
                continue;
            end
            
            gini = gini_impurity(Y(left_indices), Y(right_indices));
            
            if gini < best_gini
                best_gini = gini;
                best_feature = feature;
                best_threshold = threshold;
                feature_gini(feature) = best_gini;
            end
        end
    end
    
end
%4.gini_impurity函数
%计算Gini系数，基尼指数：衡量数据集的不纯度，值越小表示纯度越高。
%计算步骤：
% 计算左右子集的样本数比例（p_left 和 p_right）。
% 分别计算左右子集的基尼指数（gini_left 和 gini_right）。
% 加权平均得到总基尼指数。
function gini = gini_impurity(left_Y, right_Y)
    a = 50;
    edges = [-1 0.5 2];

    n_left = numel(left_Y);
    n_right = numel(right_Y);
    n_total = n_left + n_right;
    
    p_left = n_left / n_total;
    p_right = n_right / n_total;

    N_left = histcounts(left_Y, edges,'Normalization', 'probability');
    N_right = histcounts(right_Y, edges,'Normalization', 'probability');

    gini_left = 2*a*N_left(1)*N_left(2)/((a*N_left(1) + N_left(2))^2);
    gini_right = 2*a*N_right(1)*N_right(2)/((a*N_right(1) + N_right(2))^2);
    
    gini = p_left * gini_left + p_right * gini_right;
end
% 5.scanDir函数定义
%函数：scanDir 遍历总文件夹下的所有文件
%函数输入：root_dir 主目录文件夹
%输出：所有的包含路径的文件名
function [ str_matrix ] = scanDir( root_dir )  
  
files={};  
if root_dir(end)~='/'  
 root_dir=[root_dir,'/'];  
end  
fileList=dir(root_dir);  %扩展名  
n=length(fileList);  
cntpic=0;  
for i=1:n  
    if strcmp(fileList(i).name,'.')==1||strcmp(fileList(i).name,'..')==1  
        continue;  
    else  
        fileList(i).name;
        if ~fileList(i).isdir  
              
            full_name=[root_dir,fileList(i).name];  
              
%              [pathstr,name,ext,versn]=fileparts(full_name);  
%              if strcmp(ext,'.jpg')  
                 cntpic=cntpic+1;  
                 files(cntpic)={full_name};  
%              end  
        else  
            files=[files,scanDir([root_dir,fileList(i).name])];  
        end  
    end  
end  
 % 获取输入矩阵的大小
[m, n] = size(files);

% 初始化输出矩阵
str_matrix = strings(m, n);

% 遍历输入矩阵的每个元素，将其转换为字符串并存储到输出矩阵中
for i = 1:m
    for j = 1:n
        % 将单个元素转换为字符串，并将其存储到对应位置的输出矩阵中
        str_matrix(i,j) = string(files{i,j});
    end
end
end 
% 6.随机森林定义
function model = random_forest_train(X, Y, numTrees, minSamplesSplit, maxDepth, numFeatures)
% 输入参数:
%   X: 训练样本特征 (N×D矩阵, N:样本数, D:特征数)
%   Y: 训练样本标签 (N×1向量, 取值0或1)
%   numTrees: 树的数量
%   minSamplesSplit: 节点分裂的最小样本数
%   maxDepth: 树的最大深度
%   numFeatures: 每棵树使用的特征数量
%
% 输出:
%   model: 训练好的随机森林模型

[N, D] = size(X);
model.numTrees = numTrees;
model.trees = cell(numTrees, 1);  % 存储所有决策树
model.numFeatures = numFeatures;  % 记录特征数，预测时使用

% 训练每棵决策树
for i = 1:numTrees
    % 1. Bootstrap抽样 (有放回抽样)
    sampleIdx = randsample(N, N, true);
    X_train = X(sampleIdx, :);
    Y_train = Y(sampleIdx);
    
    % 2. 为当前树随机选择特征子集
    featureIdx = randperm(D, numFeatures);
    
    % 3. 训练单棵CART树
    tree = build_tree(X_train(:, featureIdx), Y_train, minSamplesSplit, maxDepth, 0);
    
    % 存储树和特征索引
    model.trees{i} = tree;
    model.featureIdx{i} = featureIdx;  % 记录该树使用的特征索引
end
end

% 7.递归构建CART决策树
function tree = build_tree(X, Y, minSamplesSplit, maxDepth, currentDepth)
% 终止条件:
%   a. 所有样本属于同一类别
%   b. 样本数小于minSamplesSplit
%   c. 达到最大深度
uniqueClasses = unique(Y);
if numel(uniqueClasses) == 1 || size(X, 1) < minSamplesSplit || currentDepth >= maxDepth
    tree.is_leaf = true;
    tree.class = mode(Y);  % 返回多数类作为预测
    return;
end

% 8.寻找最佳分裂点
[best_feature, best_value, best_gini] = find_best_split(X, Y);

% 如果无法找到有效分裂，则创建叶子节点
if best_feature == -1
    tree.is_leaf = true;
    tree.class = mode(Y);
    return;
end

% 递归构建子树
left_idx = X(:, best_feature) <= best_value;
right_idx = ~left_idx;

tree.is_leaf = false;
tree.split_feature = best_feature;
tree.split_value = best_value;
tree.left = build_tree(X(left_idx, :), Y(left_idx), minSamplesSplit, maxDepth, currentDepth + 1);
tree.right = build_tree(X(right_idx, :), Y(right_idx), minSamplesSplit, maxDepth, currentDepth + 1);
end

% 9.寻找最佳分裂特征和阈值
function [best_feature, best_value, best_gini] = find_best_split(X, Y)
[N, D] = size(X);
best_gini = inf;
best_feature = -1;
best_value = -1;

for d = 1:D
    values = unique(X(:, d));
    for i = 1:length(values)
        threshold = values(i);
        left_idx = X(:, d) <= threshold;
        right_idx = ~left_idx;
        
        % 跳过无效分裂
        if sum(left_idx) == 0 || sum(right_idx) == 0
            continue;
        end
        
        % 计算基尼指数
        gini = gini_impurity(Y(left_idx), Y(right_idx));
        
        % 更新最佳分裂
        if gini < best_gini
            best_gini = gini;
            best_feature = d;
            best_value = threshold;
        end
    end
end
end
% 10.随机森林预测函数
function predictions = random_forest_predict(model, X)
% 输入:
%   model: 训练好的随机森林模型
%   X: 测试样本特征 (M×D矩阵)
%
% 输出:
%   predictions: 预测标签 (M×1向量)

[M, ~] = size(X);
votes = zeros(M, model.numTrees);  % 存储每棵树的投票结果

% 遍历所有树进行预测
for i = 1:model.numTrees
    tree = model.trees{i};
    featureIdx = model.featureIdx{i};  % 该树使用的特征索引
    X_sub = X(:, featureIdx);         % 选择对应特征
    
    % 对每个样本进行预测
    for j = 1:M
        votes(j, i) = predict_tree(tree, X_sub(j, :));
    end
end

% 多数投票确定最终预测
predictions = mode(votes, 2);
end

% 11.单棵树预测函数
function pred = predict_tree(tree, sample)
% 递归遍历树直到叶子节点
while ~tree.is_leaf
    if sample(tree.split_feature) <= tree.split_value
        tree = tree.left;
    else
        tree = tree.right;
    end
end
pred = tree.class;
end