clc
clear 
fs=12000;%采样频率
Ts=1/fs;%采样周期
L=1500;%采样点数
t=(0:L-1)*Ts;%时间序列
STA=1; %采样起始位置
%----------------导入内圈故障的数据-----------------------------------------
load 12k_Drive_End_IR007_0_105.mat
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

  %----------------------计算中心频率确定分解个数K-----------------------------
average=mean(omega);  %对omega求平均值，即为中心频率。

figure(2)
for i = 1:K
	Hy(i,:)= abs(hilbert(u(i,:)));
    subplot(K,1,i);
    plot(t,u(i,:),'k',t,Hy(i,:),'r');
     xlabel('样点'); ylabel('幅值')
     grid; legend('信号','包络'); 
end
title('信号和包络');
set(gcf,'color','w');

% 画包络谱
figure('Name','包络谱','Color','white');
nfft=fix(L/2); 
for i = 1:K
    p=abs(fft(Hy(i,:))); %并fft，得到p，就是包络线的fft---包络谱 
    p = p/length(p)*2;
    p = p(1: fix(length(p)/2));
    subplot(K,1,i);
    plot((0:nfft-1)/nfft*fs/2,p)   %绘制包络谱
    %xlim([0 600]) %展示包络谱低频段，这句代码可以自己根据情况选择是否注释
     if i ==1
    title('包络谱'); xlabel('频率'); ylabel('幅值')
     else
        xlabel('频率'); ylabel('幅值')
     end
end
set(gcf,'color','w');

%% 计算峭度值
for i=1:K
  a(i)=kurtosis(imfn(i,:));%峭度
  disp(['IMF',num2str(i),'的峭度值为：',num2str(a(i))])
end
figure
b = bar(a,0.3);
xlabel('模态函数'); ylabel('峭度值')
set(gca,'xtick',1:1:K);
set(gca,'xticklabel',{'IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8'});
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints; %获取Bar对象的XEndPoints和YEndPoints属性
labels1 = string(b.YData); %获取条形末端坐标
text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
%指定垂直、水平对其，让值显示在条形末端居中

%% 能量熵
for i=1:K
   Eimf(i) = sum(imfn(i,:).^2,2);
end
disp(['IMF分量的能量'])
disp(Eimf(1:K))
% 能量熵
E = sum(Eimf);
for j = 1:K
    p(j) = Eimf(j)/E;
    HE(j)=-sum(p(j).*log(p(j)));
end
disp('EMD能量熵=%.4f');
disp(HE(1:K));



for i = 1:K
	xx= abs(hilbert(u(i,:))); %最小包络熵计算公式！
	xxx = xx/sum(xx);
    ssum=0;
	for ii = 1:size(xxx,2)
		bb = xxx(1,ii)*log(xxx(1,ii));
        ssum=ssum+bb;
    end
    Enen(i,:) = -ssum;%每个IMF分量的包络熵
    disp(['IMF',num2str(i),'的包络熵为：',num2str(Enen(i,:))])
end
ff = min(Enen);%求取局部最小包络熵，一般用智能优化算法优化VMD，最为最小适应度函数使用
disp(['局部最小包络熵为：',num2str(ff)])

figure('Name','频谱图','Color','white');
for i = 1:K
    p=abs(fft(u(i,:))); %并fft，得到p，就是包络线的fft---包络谱 
    subplot(K,1,i);
    plot((0:L-1)*fs/L,p)   %绘制包络谱
    xlim([0 fs/2]) %展示包络谱低频段，这句代码可以自己根据情况选择是否注释
     if i ==1
    title('频谱图'); xlabel('频率'); ylabel(['IMF' int2str(i)]);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
     else
        xlabel('频率');  ylabel(['IMF' int2str(i)]);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
     end
end
set(gcf,'color','w');
