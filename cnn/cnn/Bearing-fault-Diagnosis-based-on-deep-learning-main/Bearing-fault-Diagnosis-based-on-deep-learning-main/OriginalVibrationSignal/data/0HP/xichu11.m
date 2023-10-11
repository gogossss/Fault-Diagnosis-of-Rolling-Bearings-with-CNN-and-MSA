tic
clc
clear all
 
load('normal_0_97.mat')
x=X097_DE_time;
t=1:length(X097_DE_time);
 
 
%--------- 对于VMD参数进行设置---------------
alpha = 2000;       % moderate bandwidth constraint：适度的带宽约束/惩罚因子
tau = 0.0244;          % noise-tolerance (no strict fidelity enforcement)：噪声容限（没有严格的保真度执行）
K = 7;              % modes：分解的模态数
DC = 0;             % no DC part imposed：无直流部分
init = 1;           % initialize omegas uniformly  ：omegas的均匀初始化
tol = 1e-6 ;        
%--------------- Run actual VMD code:数据进行vmd分解---------------------------
[u, u_hat, omega] = VMD(x, alpha, tau, K, DC, init, tol);
figure;
imfn=u;
n=size(imfn,1); %size(X,1),返回矩阵X的行数；size(X,2),返回矩阵X的列数；N=size(X,2)，就是把矩阵X的列数赋值给N
 
 
 
 
for n1=1:n
    subplot(n,1,n1);
    plot(t,u(n1,:));%输出IMF分量，a(:,n)则表示矩阵a的第n列元素，u(n1,:)表示矩阵u的n1行元素
    ylabel(['IMF' ,int2str(n1)],'fontsize',11);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
end
 xlabel('样本序列','fontsize',14,'fontname','宋体');
 
%时间\itt/s
 toc;
 



