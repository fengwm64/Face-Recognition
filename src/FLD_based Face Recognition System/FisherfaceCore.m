function [m_database, V_PCA, V_Fisher, ProjectedImages_Fisher] = FisherfaceCore(T,Class_number,Class_population)
% 使用主成分分析（PCA）和Fisher线性判别（FLD）来确定人脸图像之间最具区分性的特征。
%
% 说明: 此函数获取包含所有训练图像向量的2D矩阵，并返回从训练数据库中提取的4个输出。
% 假设Ti是一张训练图像，已被重塑为1D向量。P是MxN训练图像的总数，C是类别数。
% 首先，中心化的Ti通过V_PCA传递矩阵映射到(P-C)线性子空间：Zi = V_PCA * (Ti - m_database)。
% 然后，Zi通过投影到(C-1)线性子空间转换为Yi，使同一类（或人）的图像更接近，不同类的图像更远：Yi = V_Fisher' * Zi = V_Fisher' * V_PCA' * (Ti - m_database)。
%
% 参数:      T                      - (M*NxP) 一个包含所有1D图像向量的2D矩阵。
%                                         所有1D列向量的长度相同为M*N，‘T’将是一个MNxP的2D矩阵。
% 
% 返回:       m_database             - (M*Nx1) 训练数据库的均值图像
%                V_PCA                  - (M*Nx(P-C)) 训练数据库协方差矩阵的特征向量
%                V_Fisher               - ((P-C)x(C-1)) 矩阵J = inv(Sw) * Sb的最大(C-1)特征向量
%                ProjectedImages_Fisher - ((C-1)xP) 投影到Fisher线性空间的训练图像
%
         

P = Class_population * Class_number; % 训练图像的总数

%%%%%%%%%%%%%%%%%%%%%%%% 计算均值图像 
m_database = mean(T, 2); 

%%%%%%%%%%%%%%%%%%%%%%%% 计算每个图像相对于均值图像的偏差
A = T - repmat(m_database, 1, P);

%%%%%%%%%%%%%%%%%%%%%%%% 使用特征脸算法的快照方法
L = A' * A; % L是协方差矩阵C = A * A'的代理矩阵。
[V, D] = eig(L); % D的对角元素是L = A' * A和C = A * A'的特征值。

%%%%%%%%%%%%%%%%%%%%%%%% 排序并消除小特征值
L_eig_vec = [];
for i = 1 : P - Class_number 
    L_eig_vec = [L_eig_vec V(:, i)];
end

%%%%%%%%%%%%%%%%%%%%%%%% 计算协方差矩阵'C'的特征向量
V_PCA = A * L_eig_vec; % A：中心化的图像向量

%%%%%%%%%%%%%%%%%%%%%%%% 将中心化的图像向量投影到特征空间
% Zi = V_PCA' * (Ti - m_database)
ProjectedImages_PCA = [];
for i = 1 : P
    temp = V_PCA' * A(:, i);
    ProjectedImages_PCA = [ProjectedImages_PCA temp]; 
end

%%%%%%%%%%%%%%%%%%%%%%%% 计算特征空间中每个类别的均值
m_PCA = mean(ProjectedImages_PCA, 2); % 特征空间中的总均值
m = zeros(P - Class_number, Class_number); 
Sw = zeros(P - Class_number, P - Class_number); % 初始化类内散布矩阵
Sb = zeros(P - Class_number, P - Class_number); % 初始化类间散布矩阵

for i = 1 : Class_number
    m(:, i) = mean(ProjectedImages_PCA(:, ((i-1)*Class_population + 1):i*Class_population), 2);    
    
    S  = zeros(P - Class_number, P - Class_number); 
    for j = ((i-1)*Class_population + 1) : (i*Class_population)
        S = S + (ProjectedImages_PCA(:, j) - m(:, i)) * (ProjectedImages_PCA(:, j) - m(:, i))';
    end
    
    Sw = Sw + S; % 类内散布矩阵
    Sb = Sb + (m(:, i) - m_PCA) * (m(:, i) - m_PCA)'; % 类间散布矩阵
end

%%%%%%%%%%%%%%%%%%%%%%%% 计算Fisher判别基
% 我们希望最大化类间散布矩阵，同时最小化类内散布矩阵。因此，定义了一个代价函数J，以满足此条件。
[J_eig_vec, J_eig_val] = eig(Sb, Sw); % 代价函数J = inv(Sw) * Sb
J_eig_vec = fliplr(J_eig_vec);

%%%%%%%%%%%%%%%%%%%%%%%% 消除零特征值并按降序排序
V_Fisher = zeros(size(J_eig_vec, 1), Class_number - 1); % 预先分配内存
for i = 1 : Class_number - 1 
    V_Fisher(:, i) = J_eig_vec(:, i); % 矩阵J的最大(C-1)特征向量
end

%%%%%%%%%%%%%%%%%%%%%%%% 将图像投影到Fisher线性空间
% Yi = V_Fisher' * V_PCA' * (Ti - m_database)
ProjectedImages_Fisher = zeros(Class_number - 1, Class_number * Class_population); % 预先分配内存
for i = 1 : Class_number * Class_population
    ProjectedImages_Fisher(:, i) = V_Fisher' * ProjectedImages_PCA(:, i);
end

end
