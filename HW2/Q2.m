folderpath="D:/homework/nctu/senior/fall/machine_learning/hw/hw2/dataset/Faces/s";
input1=zeros(10309,10);
input2=zeros(10309,10);
input3=zeros(10309,10);
input4=zeros(10309,10);
input5=zeros(10309,10);

for i =1:5 % number of folders 
    for j= 1: 10 %number of .pgm
        datapath = sprintf(folderpath+"%d/%d.pgm",i,j);
        datapath_str=convertStringsToChars(datapath);
        img=imread(datapath_str);
        imaged=im2double(img);%normalize image 
        if i==1
            input1(:,j)=[reshape(imaged,[],1);1;0;0;0;0];
        elseif i==2
            input2(:,j)=[reshape(imaged,[],1);0;1;0;0;0];
        elseif i==3
            input3(:,j)=[reshape(imaged,[],1);0;0;1;0;0];
        elseif i==4
            input4(:,j)=[reshape(imaged,[],1);0;0;0;1;0];
        else
            input5(:,j)=[reshape(imaged,[],1);0;0;0;0;1];
        end
    end
end
%random column
cols = size(input1,2);
P = randperm(cols);
input1_rand = input1(:,P);
%-----------------------------------
cols = size(input2,2);
P = randperm(cols);
input2_rand = input2(:,P);
%-----------------------------------
cols = size(input3,2);
P = randperm(cols);
input3_rand = input3(:,P);
%-----------------------------------
cols = size(input4,2);
P = randperm(cols);
input4_rand = input4(:,P);
%-----------------------------------
cols = size(input5,2);
P = randperm(cols);
input5_rand = input5(:,P);

%test and train data
test=[input1_rand(:,1:5),input2_rand(:,1:5),input3_rand(:,1:5),input4_rand(:,1:5),input5_rand(:,1:5)];
train=[input1_rand(:,6:10),input2_rand(:,6:10),input3_rand(:,6:10),input4_rand(:,6:10),input5_rand(:,6:10)];
% %test and train data random permutation
% cols = size(test,2);
% P = randperm(cols);
% test_rand = test(:,P);
% %-------------------------
% cols = size(train,2);
% P = randperm(cols);
% train_rand = train(:,P);

%initial data
phi_train=train(1:10304,:);
t_train=train(10305:10309,:);
[M_t_train,index_t_train] = max(t_train);
phi_test=test(1:10304,:);
t_test=test(10305:10309,:);
[M_t_test,index_t_test] = max(t_test);
w=zeros(5,10304);
de_error=zeros(5,10304);
num=0;
epoch=100;
error=zeros(epoch:1);
accuracy=zeros(epoch:1);
H=0.001;
for epochs = 1:epoch
    a=w*phi_train;
    y=exp(a)./[sum(exp(a));sum(exp(a));sum(exp(a));sum(exp(a));sum(exp(a))];
    error(epochs,:)=sum(sum(-t_train.*log(y)),2);
    de_error=phi_train*(y-t_train)';
    w=w-H*de_error';
    %accuracy result
    [M,index] = max(y);
    accuracy_matrix = index==index_t_train;
    accuracy(epochs,:)=sum(accuracy_matrix,2)/25.0;
end
%plot figure
figure
plot(error);
title('Learning curve of E(w)');
xlabel('Number of epochs') 
ylabel('Error') 
savepath_str = sprintf("D:/homework/nctu/senior/fall/machine_learning/hw/hw2/result_1/2/error.png");
savepath_chr=convertStringsToChars(savepath_str);
saveas(gcf,savepath_chr);
%plot figure
figure
plot(accuracy);
title('Accuracy of classification');
xlabel('Number of epochs') 
ylabel('Accuracy')
savepath_str = sprintf("D:/homework/nctu/senior/fall/machine_learning/hw/hw2/result_1/2/accuracy.png");
savepath_chr=convertStringsToChars(savepath_str);
saveas(gcf,savepath_chr); 

%Show the classification result of test data.
a=w*phi_test;
y_test=exp(a)./[sum(exp(a));sum(exp(a));sum(exp(a));sum(exp(a));sum(exp(a))];
error_test=sum(sum(-t_test.*log(y_test)),2);
%accuracy result
[M_test,index_test] = max(y_test);
classifiaction_result=zeros(5,25);
for z=1:25
    for z1=1:5
        if z1==index_test(:,z)
            classifiaction_result(z1,z)=1; 
        end
    end
end    
accuracy_matrix_test = index_test==index_t_test;
accuracy_test=sum(accuracy_matrix_test,2)/25.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%run Q2_PCA.m
