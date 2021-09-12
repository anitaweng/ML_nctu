%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%run Q2_2_pca.m first
face_img_dim=10;%PCA should be used to reduce the dimension of face images to 2, 5 and 10
pca_image_filter=Vs(:,1:face_img_dim);

% new_X=pca_image_filter'*X';
input1_pca=pca_image_filter'*input1(1:10304,:);
input1_pca=[input1_pca;ones(1,10);zeros(1,10);zeros(1,10);zeros(1,10);zeros(1,10)];
%------------------------------------------------------------------------------------
input2_pca=pca_image_filter'*input2(1:10304,:);
input2_pca=[input2_pca;zeros(1,10);ones(1,10);zeros(1,10);zeros(1,10);zeros(1,10)];
%------------------------------------------------------------------------------------
input3_pca=pca_image_filter'*input3(1:10304,:);
input3_pca=[input3_pca;zeros(1,10);zeros(1,10);ones(1,10);zeros(1,10);zeros(1,10)];
%------------------------------------------------------------------------------------
input4_pca=pca_image_filter'*input4(1:10304,:);
input4_pca=[input4_pca;zeros(1,10);zeros(1,10);zeros(1,10);ones(1,10);zeros(1,10)];
%------------------------------------------------------------------------------------
input5_pca=pca_image_filter'*input5(1:10304,:);
input5_pca=[input5_pca;zeros(1,10);zeros(1,10);zeros(1,10);zeros(1,10);ones(1,10)];

%random column
cols = size(input1_pca,2);
P = randperm(cols);
input1_rand = input1_pca(:,P);
%-----------------------------------
cols = size(input2_pca,2);
P = randperm(cols);
input2_rand = input2_pca(:,P);
%-----------------------------------
cols = size(input3_pca,2);
P = randperm(cols);
input3_rand = input3_pca(:,P);
%-----------------------------------
cols = size(input4_pca,2);
P = randperm(cols);
input4_rand = input4_pca(:,P);
%-----------------------------------
cols = size(input5_pca,2);
P = randperm(cols);
input5_rand = input5_pca(:,P);

%test and train data
test=[input1_rand(:,1:5),input2_rand(:,1:5),input3_rand(:,1:5),input4_rand(:,1:5),input5_rand(:,1:5)];
train=[input1_rand(:,6:10),input2_rand(:,6:10),input3_rand(:,6:10),input4_rand(:,6:10),input5_rand(:,6:10)];

%initial data
phi_train=train(1:face_img_dim,:);
t_train=train((face_img_dim+1):(face_img_dim+5),:);
[M_t_train,index_t_train] = max(t_train);
phi_test=test(1:face_img_dim,:);
t_test=test((face_img_dim+1):(face_img_dim+5),:);
[M_t_test,index_t_test] = max(t_test);
w=zeros(5,face_img_dim);
de_error=zeros(5,face_img_dim);
num=0;
epoch=100;%   2=5000   5=2000    10=100/1000
error=zeros(epoch:1);
accuracy=zeros(epoch:1);
for epochs = 1:epoch
    a=w*phi_train;
    y=exp(a)./[sum(exp(a));sum(exp(a));sum(exp(a));sum(exp(a));sum(exp(a))];
    error(epochs,:)=sum(sum(-t_train.*log(y)),2);
    de_error=phi_train*(y-t_train)';
    y_mul=y.*(1.-y);
    phi_times=phi_train*phi_train';
%     H=zeros(face_img_dim,face_img_dim);
    for i=1:5
        H=zeros(face_img_dim,face_img_dim);
        for j=1:25
            H=H+(y_mul(i,j).*phi_times);
        end
        w(i,:)=w(i,:)-de_error(:,i)'*pinv(H');
    end
    
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
savepath_str = sprintf("D:/homework/nctu/senior/fall/machine_learning/hw/hw2/result_1/2/error"+int2str(face_img_dim)+".png");
savepath_chr=convertStringsToChars(savepath_str);
saveas(gcf,savepath_chr);
%plot figure
figure
plot(accuracy);
title('Accuracy of classification');
xlabel('Number of epochs') 
ylabel('Accuracy')
savepath_str = sprintf("D:/homework/nctu/senior/fall/machine_learning/hw/hw2/result_1/2/accuracy"+int2str(face_img_dim)+".png");
savepath_chr=convertStringsToChars(savepath_str);
saveas(gcf,savepath_chr); 

%Show the classification result of test data.
a=w*phi_test;
y_test=exp(a)./[sum(exp(a));sum(exp(a));sum(exp(a));sum(exp(a));sum(exp(a))];
error_test=sum(sum(-t_test.*log(y_test)),2);%   2=8.746426752437445    5=0.399496471289282  10=0.256974773638008(80)    10=0.180219150118358(1000)
%accuracy result
[M_test,index_test] = max(y_test);
accuracy_matrix_test = index_test==index_t_test;
accuracy_test=sum(accuracy_matrix_test,2)/25.0;%   2=0.76   5=1   10=1(80)   10=1(1000)


