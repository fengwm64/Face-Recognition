% 一个示例脚本，展示了PCA（基于主成分分析）人脸识别系统中的函数用法（Eigenface方法）
%
% 另见: CREATEDATABASE, EIGENFACECORE, RECOGNITION           

clear all % 清除所有变量
clc % 清除命令行
close all % 关闭所有图形窗口
%% 

% 你可以自定义和固定初始目录路径
TrainDatabasePath = uigetdir('F:\OneDrive\课程资料\计算机视觉与模式识别\实验\实验5\实验5参考代码\PCA_based Face Recognition System\TrainDatabase', '设置训练图片所处文件夹路径' );
% 选择训练数据库路径
TestDatabasePath = uigetdir('F:\OneDrive\课程资料\计算机视觉与模式识别\实验\实验5\实验5参考代码\PCA_based Face Recognition System\TestDatabase', '设置测试图片所处文件夹路径');
% 选择测试数据库路径

% 输入测试图像的名称
prompt = {'Enter test image name (a number between 1 to 10):'};
dlg_title = 'Input of PCA-Based Face Recognition System';
num_lines= 1;
def = {'1'};

% 获取用户输入的测试图像编号
TestImage  = inputdlg(prompt,dlg_title,num_lines,def);
TestImage = strcat(TestDatabasePath,'\',char(TestImage),'.jpg'); % 构建测试图像路径
im = imread(TestImage); % 读取测试图像

% 创建数据库
T = CreateDatabase(TrainDatabasePath);
% 计算特征脸
[m, A, Eigenfaces] = EigenfaceCore(T);
% 识别测试图像
OutputName = Recognition(TestImage, m, A, Eigenfaces);

% 读取识别的图像
SelectedImage = strcat(TrainDatabasePath,'\',OutputName);
SelectedImage = imread(SelectedImage);

% 显示测试图像
imshow(im)
title('Test Image');
% 显示识别的图像
figure,imshow(SelectedImage);
title('Equivalent Image');

% 输出匹配的图像名称
str = strcat('Matched image is :  ',OutputName);
disp(str)
