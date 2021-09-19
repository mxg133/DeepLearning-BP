%% 初始化
clear
close all
clc
format short
%% 读取读取
data=xlsread('数据.xlsx','Sheet1','G1:J2000'); %%使用xlsread函数读取EXCEL中对应范围的数据即可  

%输入输出数据
input=data(:,1:end-1);    %data的第一列-倒数第二列为特征指标
output=data(:,end);  %data的最后面一列为输出的指标值

N=length(output);   %全部样本数目
testNum=200;   %设定测试样本数目
trainNum=N-testNum;    %计算训练样本数目

%% 划分训练集、测试集
input_train = input(1:trainNum,:)';
output_train =output(1:trainNum)';
input_test =input(trainNum+1:trainNum+testNum,:)';
output_test =output(trainNum+1:trainNum+testNum)';

%% 数据归一化
[inputn,inputps]=mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train);
inputn_test=mapminmax('apply',input_test,inputps);

%% 获取输入层节点、输出层节点个数
inputnum=size(input,2);
outputnum=size(output,2);
disp('/////////////////////////////////')
disp('神经网络结构...')
disp(['输入层的节点数为：',num2str(inputnum)])
disp(['输出层的节点数为：',num2str(outputnum)])
disp(' ')
disp('隐含层节点的确定过程...')

%确定隐含层节点个数
%采用经验公式hiddennum=sqrt(m+n)+a，m为输入层节点个数，n为输出层节点个数，a一般取为1-10之间的整数
MSE=1e+5; %初始化最小误差
transform_func={'tansig','purelin'}; %激活函数
train_func='trainlm';   %训练算法
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    %构建网络
    net=newff(inputn,outputn,hiddennum,transform_func,train_func);
    % 网络参数
    net.trainParam.epochs=100000000000;         % 训练次数
    net.trainParam.lr=0.00000000000000000001;                   % 学习速率
    net.trainParam.goal=0.0000000000001;        % 训练目标最小误差
    % 网络训练
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);  %仿真结果
    mse0=mse(outputn,an0);  %仿真的均方误差
    disp(['隐含层节点数为',num2str(hiddennum),'时，训练集的均方误差为：',num2str(mse0)])
    
    %更新最佳的隐含层节点
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['最佳的隐含层节点数为：',num2str(hiddennum_best),'，相应的均方误差为：',num2str(MSE)])

%% 构建最佳隐含层节点的BP神经网络
net=newff(inputn,outputn,hiddennum_best,transform_func,train_func);

% 网络参数
net.trainParam.epochs=100000000000;         % 训练次数
net.trainParam.lr=0.51;                   % 学习速率
net.trainParam.goal=0.000001;        % 训练目标最小误差

%% 网络训练
net=train(net,inputn,outputn);

%% 网络测试
an=sim(net,inputn_test); %用训练好的模型进行仿真
test_simu=mapminmax('reverse',an,outputps); % 预测结果反归一化

error=test_simu-output_test;      %预测值和真实值的误差

%%真实值与预测值误差比较
figure
plot(output_test,'bo-','linewidth',1.2)
hold on
plot(test_simu,'r*-','linewidth',1.2)
legend('实际分析数据','预测测试数据')
xlabel('测试样本编号'),ylabel('发动机健康状态预测(h)')
%title('BP测试集预测值和期望值的对比')
set(gca,'fontsize',12)

figure
plot(error,'ro-','linewidth',1.2)
xlabel('测试样本编号'),ylabel('预测偏差')
title('BP神经网络测试集的预测误差')
set(gca,'fontsize',12)

%计算误差
[~,len]=size(output_test);
SSE1=sum(error.^2);
MAE1=sum(abs(error))/len;
MSE1=error*error'/len;
RMSE1=MSE1^(1/2);
MAPE1=mean(abs(error./output_test));
r=corrcoef(output_test,test_simu);    %corrcoef计算相关系数矩阵，包括自相关和互相关系数
R1=r(1,2);    

disp(' ')
disp('/////////////////////////////////')
disp('预测误差分析...')
disp(['误差平方和SSE为：            ',num2str(SSE1)])
disp(['平均绝对误差MAE为：      ',num2str(MAE1)])
disp(['均方误差MSE为：              ',num2str(MSE1)])
disp(['均方根误差RMSE为：        ',num2str(RMSE1)])
disp(['平均百分比误差MAPE为： ',num2str(MAPE1*100),'%'])
disp(['相关系数R为：                     ',num2str(R1)])

%打印结果
disp(' ')
disp('/////////////////////////////////')
disp('打印测试集预测结果...')
disp(['    编号         实际值        预测值        误差'])
for i=1:len
    disp([i,output_test(i),test_simu(i),error(i)])
end





