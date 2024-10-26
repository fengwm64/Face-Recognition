function [m, A, Eigenfaces] = EigenfaceCore(T)
% 使用主成分分析(PCA)来确定人脸图像之间最具区分性的特征。
%
% 描述: 该函数接收一个包含所有训练图像向量的2D矩阵，
% 并返回从训练数据库中提取的3个输出。
%
% 参数:       T                      - 一个2D矩阵，包含所有1D图像向量。
%                                       假设训练数据库中的所有P张图像尺寸相同，为MxN。
%                                       那么1D列向量的长度为M*N，矩阵'T'的尺寸为MNxP。
% 
% 返回值:     m                      - (M*Nx1) 训练数据库的均值
%             Eigenfaces             - (M*Nx(P-1)) 训练数据库协方差矩阵的特征向量
%             A                      - (M*NxP) 居中的图像向量矩阵
% 

%%%%%%%%%%%%%%%%%%%%%%%% 计算均值图像 %%%%%%%%%%%%%%%%%%%%%%%%
m = mean(T, 2);            % 计算平均人脸图像 m = (1/P)*sum(Tj's)    (j = 1 : P)

%%%%%%%%%%%%%%%%%%%%%%%% 计算每张图像与均值图像的偏差 %%%%%%%%%%%%%%%%%%%%%%%%
A = bsxfun(@minus, double(T), m);

%%%%%%%%%%%%%%%%%%%%%%%% 特征脸方法的快照方法 %%%%%%%%%%%%%%%%%%%%%%%%
% 根据线性代数理论，对于一个PxQ矩阵，最大非零特征值的数量是min(P-1,Q-1)。
% 由于训练图像的数量(P)通常小于像素数(M*N)，可以找到的最大非零特征值数量等于P-1。
% 因此我们可以计算A'*A (一个PxP矩阵)的特征值，而不是A*A' (一个M*NxM*N矩阵)的特征值。
% 显然，A*A'的维数远大于A'*A。所以维数会降低。

L = A' * A;         % L是协方差矩阵C=A*A'的替代矩阵。
[V, D] = eig(L);    % D的对角元素是L=A'*A和C=A*A'的特征值。

%%%%%%%%%%%%%%%%%%%%%%%% 排序并消除特征值 %%%%%%%%%%%%%%%%%%%%%%%%
% 对矩阵L的所有特征值进行排序，消除那些小于指定阈值的特征值。
% 所以非零特征向量的数量可能少于(P-1)。

eig_vals = diag(D);               % 提取特征值
valid_eigen_indices = eig_vals > 1; % 只保留特征值大于1的特征向量
L_eig_vec = V(:, valid_eigen_indices);

%%%%%%%%%%%%%%%%%%%%%%%% 计算协方差矩阵'C'的特征向量 %%%%%%%%%%%%%%%%%%%%%%%%
% 协方差矩阵C的特征向量（或称为"特征脸"）
% 可以从L的特征向量中恢复。
Eigenfaces = A * L_eig_vec; % A: 居中的图像向量

end
