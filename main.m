%% ��ʼ��
clear
close all
clc
format short
%% ��ȡ��ȡ
data=xlsread('����.xlsx','Sheet1','G1:J2000'); %%ʹ��xlsread������ȡEXCEL�ж�Ӧ��Χ�����ݼ���  

%�����������
input=data(:,1:end-1);    %data�ĵ�һ��-�����ڶ���Ϊ����ָ��
output=data(:,end);  %data�������һ��Ϊ�����ָ��ֵ

N=length(output);   %ȫ��������Ŀ
testNum=200;   %�趨����������Ŀ
trainNum=N-testNum;    %����ѵ��������Ŀ

%% ����ѵ���������Լ�
input_train = input(1:trainNum,:)';
output_train =output(1:trainNum)';
input_test =input(trainNum+1:trainNum+testNum,:)';
output_test =output(trainNum+1:trainNum+testNum)';

%% ���ݹ�һ��
[inputn,inputps]=mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train);
inputn_test=mapminmax('apply',input_test,inputps);

%% ��ȡ�����ڵ㡢�����ڵ����
inputnum=size(input,2);
outputnum=size(output,2);
disp('/////////////////////////////////')
disp('������ṹ...')
disp(['�����Ľڵ���Ϊ��',num2str(inputnum)])
disp(['�����Ľڵ���Ϊ��',num2str(outputnum)])
disp(' ')
disp('������ڵ��ȷ������...')

%ȷ��������ڵ����
%���þ��鹫ʽhiddennum=sqrt(m+n)+a��mΪ�����ڵ������nΪ�����ڵ������aһ��ȡΪ1-10֮�������
MSE=1e+5; %��ʼ����С���
transform_func={'tansig','purelin'}; %�����
train_func='trainlm';   %ѵ���㷨
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    %��������
    net=newff(inputn,outputn,hiddennum,transform_func,train_func);
    % �������
    net.trainParam.epochs=100000000000;         % ѵ������
    net.trainParam.lr=0.00000000000000000001;                   % ѧϰ����
    net.trainParam.goal=0.0000000000001;        % ѵ��Ŀ����С���
    % ����ѵ��
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);  %������
    mse0=mse(outputn,an0);  %����ľ������
    disp(['������ڵ���Ϊ',num2str(hiddennum),'ʱ��ѵ�����ľ������Ϊ��',num2str(mse0)])
    
    %������ѵ�������ڵ�
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['��ѵ�������ڵ���Ϊ��',num2str(hiddennum_best),'����Ӧ�ľ������Ϊ��',num2str(MSE)])

%% �������������ڵ��BP������
net=newff(inputn,outputn,hiddennum_best,transform_func,train_func);

% �������
net.trainParam.epochs=100000000000;         % ѵ������
net.trainParam.lr=0.51;                   % ѧϰ����
net.trainParam.goal=0.000001;        % ѵ��Ŀ����С���

%% ����ѵ��
net=train(net,inputn,outputn);

%% �������
an=sim(net,inputn_test); %��ѵ���õ�ģ�ͽ��з���
test_simu=mapminmax('reverse',an,outputps); % Ԥ��������һ��

error=test_simu-output_test;      %Ԥ��ֵ����ʵֵ�����

%%��ʵֵ��Ԥ��ֵ���Ƚ�
figure
plot(output_test,'bo-','linewidth',1.2)
hold on
plot(test_simu,'r*-','linewidth',1.2)
legend('ʵ�ʷ�������','Ԥ���������')
xlabel('�����������'),ylabel('����������״̬Ԥ��(h)')
%title('BP���Լ�Ԥ��ֵ������ֵ�ĶԱ�')
set(gca,'fontsize',12)

figure
plot(error,'ro-','linewidth',1.2)
xlabel('�����������'),ylabel('Ԥ��ƫ��')
title('BP��������Լ���Ԥ�����')
set(gca,'fontsize',12)

%�������
[~,len]=size(output_test);
SSE1=sum(error.^2);
MAE1=sum(abs(error))/len;
MSE1=error*error'/len;
RMSE1=MSE1^(1/2);
MAPE1=mean(abs(error./output_test));
r=corrcoef(output_test,test_simu);    %corrcoef�������ϵ�����󣬰�������غͻ����ϵ��
R1=r(1,2);    

disp(' ')
disp('/////////////////////////////////')
disp('Ԥ��������...')
disp(['���ƽ����SSEΪ��            ',num2str(SSE1)])
disp(['ƽ���������MAEΪ��      ',num2str(MAE1)])
disp(['�������MSEΪ��              ',num2str(MSE1)])
disp(['���������RMSEΪ��        ',num2str(RMSE1)])
disp(['ƽ���ٷֱ����MAPEΪ�� ',num2str(MAPE1*100),'%'])
disp(['���ϵ��RΪ��                     ',num2str(R1)])

%��ӡ���
disp(' ')
disp('/////////////////////////////////')
disp('��ӡ���Լ�Ԥ����...')
disp(['    ���         ʵ��ֵ        Ԥ��ֵ        ���'])
for i=1:len
    disp([i,output_test(i),test_simu(i),error(i)])
end





