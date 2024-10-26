%% Fisherface 线性判别分析人脸识别
%% 
% 

clear all
clc
close all
%%
% 选择训练数据库路径
TrainDatabasePath = uigetdir('..\人脸采集\LDA_dataset\train_dataset', '设置训练图片所处文件夹路径' );
% 选择测试数据库路径
TestDatabasePath = uigetdir('..\人脸采集\LDA_dataset\test_dataset', '设置测试图片所处文件夹路径');
%%
% 创建数据库
tic
[T,T_idx] = CreateDatabase(TrainDatabasePath);
toc

% 计算特征脸
tic
[m, V_PCA, V_Fisher, ProjectedImages_Fisher] = FisherfaceCore(T,5,200);
toc
%% 1.测试

% 获取测试数据集下的子文件夹名称
test_subfolders = dir(TestDatabasePath);
test_subfolders = test_subfolders([test_subfolders.isdir]); % 仅保留文件夹
test_subfolders = test_subfolders(~ismember({test_subfolders.name}, {'.', '..'})); % 去除当前和上级目录
subfolder_names = {test_subfolders.name};

% 从列表对话框中选择子文件夹
[selected_index, ~] = listdlg('PromptString', '选择一个子文件夹：', 'SelectionMode', 'single', 'ListString', subfolder_names);

% 获取用户选择的子文件夹路径
selected_testFolder = fullfile(TestDatabasePath, subfolder_names{selected_index});

% 获取文件夹中所有图像文件
imageFiles = dir(fullfile(selected_testFolder, '*.jpg'));

% 随机选择一张图像
randomIndex = randi(numel(imageFiles));
randomImageName = imageFiles(randomIndex).name;
randomImagePath = fullfile(selected_testFolder, randomImageName);

% 读取随机选择的图像
TestImage = imread(randomImagePath);

figure
imshow(TestImage)
title('测试图片');
%%
Selected = Recognition(TestImage, m, V_PCA, V_Fisher, ProjectedImages_Fisher);

% 显示测试图像
figure
subplot(1,2,1)
imshow(TestImage)
title('测试图片');
subplot(1,2,2)
imshow(imread(T_idx{Selected}));
title('Fisherface匹配到的图片');
%% 2.自动化验证
% 2.1 并行每个类别评测

% 获取测试数据集下的子文件夹名称
test_subfolders = dir(TestDatabasePath);
test_subfolders = test_subfolders([test_subfolders.isdir]); % 仅保留文件夹
test_subfolders = test_subfolders(~ismember({test_subfolders.name}, {'.', '..'})); % 去除当前和上级目录
subfolder_names = {test_subfolders.name};

% 遍历测试数据集的每个子文件夹
% 初始化评价结果变量
evaluation_results = cell(1, numel(subfolder_names));

% 使用 parfor 循环并行处理每个子文件夹
tic
parfor i = 1:numel(subfolder_names)
    folder_name = subfolder_names{i};
    folder_path = fullfile(TestDatabasePath, folder_name);
    
    % 获取子文件夹中所有图像文件
    image_files = dir(fullfile(folder_path, '*.jpg'));
    image_names = {image_files.name};
    
    % 初始化当前类别的评价结果
    folder_evaluation_results = struct();
    folder_evaluation_results.folder_name = folder_name;
    folder_evaluation_results.image_names = image_names;
    folder_evaluation_results.recognition_results = cell(size(image_names));
    folder_evaluation_results.recognition_class = cell(size(image_names));

    % 遍历当前子文件夹下的每张图像
    for j = 1:numel(image_names)
        % 读取图像
        image_path = fullfile(folder_path, image_names{j});
        test_img = imread(image_path);
        
        % 识别图像
        Selected = Recognition(test_img, m, V_PCA, V_Fisher, ProjectedImages_Fisher);
        recognized_image_name = T_idx{Selected};
        
        % 将识别结果添加到当前类别的评价结果中
        folder_evaluation_results.recognition_results{j} = recognized_image_name;
        [~, folder_evaluation_results.recognition_class{j}, ~] = fileparts(fileparts(recognized_image_name));

        % 定期输出图像文件名
        if mod(j, 10) == 0
            fprintf('已处理图像 %d/%d\n', j, numel(image_names));
        end
    end
    
    % 将当前类别的评价结果添加到总的评价结果中
    evaluation_results{i} = folder_evaluation_results;
end
toc
% 2.2 评测结果

% 显示评价结果
for i = 1:numel(evaluation_results)
    folder_evaluation_results = evaluation_results{i};
    fprintf('评价结果 for 文件夹：%s\n', folder_evaluation_results.folder_name);
    for j = 1:numel(folder_evaluation_results.image_names)
        fprintf('图片：%s, 识别结果：%s\n', folder_evaluation_results.image_names{j}, folder_evaluation_results.recognition_results{j});
    end
    fprintf('\n\n');
end
% 2.2.1 误分类占比

% 初始化计数器
misclassified_count = zeros(size(subfolder_names));
total_count = zeros(size(subfolder_names));

% 遍历评价结果
for i = 1:numel(evaluation_results)
    folder_evaluation_results = evaluation_results{i};
    
    % 检查每张图片的识别结果是否与文件夹名称不匹配
    misclassified_indices = ~strcmp(folder_evaluation_results.folder_name, folder_evaluation_results.recognition_class);
    
    % 计算每个类别的误分类数量
    misclassified_count(i) = sum(misclassified_indices);
    total_count(i) = numel(folder_evaluation_results.image_names);
end

% 显示每个类别的误分类数量
% 显示每个类别的误分类比率
for i = 1:numel(subfolder_names)
    fprintf('文件夹 %s 误分类数量：%d/%d, 误分类比率：%.2f%%\n', subfolder_names{i}, ...
            misclassified_count(i), total_count(i), ...
            100 * misclassified_count(i) / total_count(i));
end
% 2.2.2 混淆矩阵

% 初始化混淆矩阵
num_classes = numel(subfolder_names);
confusion_matrix = zeros(num_classes, num_classes);

% 填充混淆矩阵
for i = 1:numel(evaluation_results)
    folder_evaluation_results = evaluation_results{i};
    for j = 1:numel(folder_evaluation_results.image_names)
        true_class_index = find(strcmp(subfolder_names, folder_evaluation_results.folder_name));
        predicted_class_index = find(strcmp(subfolder_names, folder_evaluation_results.recognition_class{j}));
        
        % 在混淆矩阵中增加计数
        confusion_matrix(true_class_index, predicted_class_index) = confusion_matrix(true_class_index, predicted_class_index) + 1;
    end
end

% 绘制混淆矩阵
figure;
imagesc(confusion_matrix);
colorbar;
title('混淆矩阵');
xlabel('预测类别');
ylabel('真实类别');
xticks(1:num_classes);
xticklabels(subfolder_names);
yticks(1:num_classes);
yticklabels(subfolder_names);
colormap("sky")

% 在每个格子中添加数据标签
for true_class_index = 1:num_classes
    for predicted_class_index = 1:num_classes
        text(predicted_class_index, true_class_index, num2str(confusion_matrix(true_class_index, predicted_class_index)), 'HorizontalAlignment', 'center', 'Color', 'black');
    end
end
%% 3.训练集只有一张人脸情况
% 无法运行程序，详细解释请见报告