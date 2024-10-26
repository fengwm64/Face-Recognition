function Recognized_index = Recognition(TestImage, m, A, Eigenfaces)
% 识别步骤...
%
% 参数:       TestImage              - 输入测试图像的路径
%
%             m                      - (M*Nx1) 训练数据库的均值，
%                                      由'EigenfaceCore'函数输出。
%
%             Eigenfaces             - (M*Nx(P-1)) 训练数据库协方差矩阵的
%                                      特征向量，由'EigenfaceCore'函数输出。
%
%             A                      - (M*NxP) 居中的图像向量矩阵，
%                                      由'EigenfaceCore'函数输出。
% 
% 返回值:     RecognizedImage         - 在训练数据库中识别的图像数据。
%
% 了优化上述函数的速度，可以采取以下几个措施：
% 
% 预分配内存：避免在循环中动态扩展矩阵或向量。
% 矩阵运算：尽量使用矩阵运算代替循环。
% 避免不必要的转置：减少不必要的转置操作。        

%%%%%%%%%%%%%%%%%%%%%%%% 将居中的图像向量投影到人脸空间 %%%%%%%%%%%%%%%%%%%%%%%%
% 所有居中的图像通过与特征脸基相乘投影到人脸空间。每张脸的投影向量将是其对应的
% 特征向量。

Train_Number = size(Eigenfaces,2); % 训练图像数量
ProjectedImages = Eigenfaces' * A; % 将所有居中的图像投影到人脸空间

%%%%%%%%%%%%%%%%%%%%%%%% 从测试图像中提取PCA特征 %%%%%%%%%%%%%%%%%%%%%%%%
temp = TestImage(:,:,1); % 取图像的第一个通道（灰度图像）

[irow, icol] = size(temp); % 获取图像尺寸
InImage = reshape(temp',irow*icol,1); % 将2D图像重塑为1D向量
Difference = double(InImage) - m; % 居中的测试图像
ProjectedTestImage = Eigenfaces' * Difference; % 测试图像的特征向量

%%%%%%%%%%%%%%%%%%%%%%%% 计算欧几里得距离 %%%%%%%%%%%%%%%%%%%%%%%%
% 计算投影的测试图像与所有居中的训练图像投影的欧几里得距离。
% 测试图像应该与训练数据库中的对应图像具有最小距离。

% 计算所有投影向量与测试图像投影向量之间的欧几里得距离的平方
Euc_dist = sum((ProjectedImages - ProjectedTestImage).^2, 1);

[~, Recognized_index] = min(Euc_dist); % 找到最小距离的索引

end
