
function [u, u_hat, omega] = VMD(signal, alpha, tau, K, DC, init, tol)
clc
clear 
fs=12000;%采样频率
Ts=1/fs;%采样周期
L=1500;%采样点数
t=(0:L-1)*Ts;%时间序列
STA=1; %采样起始位置
%----------------导入内圈故障的数据-----------------------------------------
load IR007_0_105.mat
X = X105_DE_time(1:L)'; %这里可以选取DE(驱动端加速度)、FE(风扇端加速度)、BA(基座加速度)，直接更改变量名，挑选一种即可。
%--------- some sample parameters forVMD：对于VMD样品参数进行设置---------------
alpha = 2500;       % moderate bandwidth constraint：适度的带宽约束/惩罚因子
tau = 0;          % noise-tolerance (no strict fidelity enforcement)：噪声容限（没有严格的保真度执行）
K = 8;              % modes：分解的模态数，可以自行设置，这里以8为例。
DC = 0;             % no DC part imposed：无直流部分
init = 1;           % initialize omegas uniformly  ：omegas的均匀初始化
tol = 1e-7;        
%--------------- Run actual VMD code:数据进行vmd分解---------------------------
[u, u_hat, omega] = VMD(X, alpha, tau, K, DC, init, tol); %其中u为分解得到的IMF分量


figure(1);
imfn=u;
n=size(imfn,1); 
subplot(n+1,1,1); 
plot(t,X); %故障信号
ylabel('原始信号','fontsize',12,'fontname','宋体');
 
for n1=1:n
    subplot(n+1,1,n1+1);
    plot(t,u(n1,:));%输出IMF分量，a(:,n)则表示矩阵a的第n列元素，u(n1,:)表示矩阵u的n1行元素
    ylabel(['IMF' int2str(n1)]);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
end
 xlabel('时间\itt/s','fontsize',12,'fontname','宋体');
