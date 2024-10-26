function Recognized_index = Recognition(TestImage, m_database, V_PCA, V_Fisher, ProjectedImages_Fisher)
% 识别步骤....
%
% 说明: 此函数通过将图像投影到特征空间并测量它们之间的欧氏距离来比较两张人脸。
%
% 参数:      TestImage              - 输入测试图像的路径
%
%                m_database             - (M*Nx1) 训练数据库的均值图像
%                                         这是 'EigenfaceCore' 函数的输出。
%
%                V_PCA                  - (M*Nx(P-1)) 训练数据库协方差矩阵的特征向量
%
%                V_Fisher               - ((P-1)x(C-1)) 矩阵 J = inv(Sw) * Sb 的最大 (C-1) 特征向量
%
%                ProjectedImages_Fisher - ((C-1)xP) 投影到 Fisher 线性空间的训练图像
% 
% 返回:       Recognized_index        - 被识别图像在训练数据库中的索引。

% 获取训练图像的数量
Train_Number = size(ProjectedImages_Fisher, 2);

%%%%%%%%%%%%%%%%%%%%%%%% 从测试图像中提取 FLD 特征
TestImage = rgb2gray(TestImage); % 将输入图像转换为灰度图像

[irow, icol] = size(TestImage); % 获取图像的行列尺寸
InImage = reshape(TestImage', irow * icol, 1); % 将二维图像重塑为一维向量
Difference = double(InImage) - m_database; % 计算中心化的测试图像
ProjectedTestImage = V_Fisher' * V_PCA' * Difference; % 计算测试图像的特征向量

%%%%%%%%%%%%%%%%%%%%%%%% 计算欧氏距离
% 计算投影的测试图像与所有中心化训练图像投影之间的欧氏距离。
% 测试图像应与训练数据库中对应的图像距离最小。

Euc_dist = zeros(1, Train_Number); % 预先分配存储欧氏距离的数组
for i = 1 : Train_Number
    q = ProjectedImages_Fisher(:, i); % 获取训练图像的特征向量
    Euc_dist(i) = (norm(ProjectedTestImage - q))^2; % 计算欧氏距离的平方并存储
end

% 找到最小的欧氏距离及其对应的索引
[~, Recognized_index] = min(Euc_dist);
end
