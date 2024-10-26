function [T, T_idx] = CreateDatabase(TrainDatabasePath)
    % 设置 HOG 参数
    cellSize = [8 8];       % 单元格大小
    blockSize = [2 2];      % 块大小
    blockOverlap = [1 1];   % 块重叠
    numBins = 9;            % 梯度方向的直方图数目

    % 获取所有子文件夹的路径
    subFolders = dir(TrainDatabasePath);
    
    % 排除 . .. 和非文件夹
    subFolders = subFolders([subFolders.isdir] & ~ismember({subFolders.name}, {'.', '..'}));
    
    % 初始化图片计数
    Train_Number = 0;
    imageFilesList = {}; % 存储所有图像文件的路径
    for i = 1:length(subFolders)
        % 获取子文件夹中的所有图片文件
        imageFiles = dir(fullfile(TrainDatabasePath, subFolders(i).name, '*.jpg'));
        imageFilesList = [imageFilesList; fullfile({subFolders(i).folder}, {subFolders(i).name}, {imageFiles.name})'];
        Train_Number = Train_Number + length(imageFiles);
    end
    
    % 读取第一张图像以获取图像尺寸
    firstImagePath = imageFilesList{1};
    img = imread(firstImagePath);
    img = rgb2gray(img);
    [features, ~] = extractHOGFeatures(img, 'CellSize', cellSize, ...
            'BlockSize', blockSize, 'BlockOverlap', blockOverlap, 'NumBins', numBins);
    
    % 初始化T和T_idx
    T = zeros(length(features), Train_Number);
    T_idx = cell(Train_Number, 1);
    
    % 并行从1D图像向量构建2D矩阵
    for imageIndex = 1:Train_Number
        % 读取图像
        img = imread(imageFilesList{imageIndex});
        img = rgb2gray(img);
        
        % 提取 HOG 特征
        [features, ~] = extractHOGFeatures(img, 'CellSize', cellSize, ...
            'BlockSize', blockSize, 'BlockOverlap', blockOverlap, 'NumBins', numBins);
        
        T(:, imageIndex) = features;

        % 构建图片路径和编号的索引
        T_idx{imageIndex} = imageFilesList{imageIndex};
    end
end
