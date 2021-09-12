%readfile
filename = 'dataset/Pokemon.csv';
[totalData,str,raw] = xlsread(filename);
x=totalData(:,4:12);
y_str=str(2:159,2);
y=zeros(158,3);
for i=1:158
    if strcmp(y_str(i,1),"Normal")
        y(i,:)=[1,0,0];
    elseif strcmp(y_str(i,1),"Psychic")
        y(i,:)=[0,1,0];
    else
        y(i,:)=[0,0,1];
    end
end    
x_train=normalize(x(1:120,:),'zscore');
% x_train=x_norm_train(1:120,:);
x_test=normalize(x(121:158,:),'zscore');
% x_test=x_norm_test;
y_train=y(1:120,:);
y_test=y(121:158,:);
%initialize 
dis=zeros(120,38);
sort_index_result=zeros(120,38);
dis_sort_result=zeros(120,38);
t_train=zeros(120,3,38);

for j=1:38
    r_x_test = repmat(x_test(j,:),120,1) ;
    V = r_x_test - x_train;
    D = V.^(2);
    dis(:,j)=sqrt(sum(D,2));%each column is about the point to the others
    [dis_sort,sort_index] = sort(dis(:,j),'ascend');
    sort_index_result(:,j)=sort_index;
    dis_sort_result(:,j)=dis_sort;
    t_train(:,:,j)=y_train(sort_index,:); 
end
result=zeros(38,10);
accuracy=zeros(1,10);
for K=1:10
    for j=1:38
        if K==1
            temp=t_train(1:K,:,j);
        else
            temp=sum(t_train(1:K,:,j));
        end
        [M,class] = max(temp);
        result(j,K)=class;
    end
    [tempy,y_test_class]=max(y_test,[],2);
    accuracy_matrix = result(:,K)==y_test_class;
    accuracy(:,K)=sum(accuracy_matrix)/38.0;
end
figure
plot(1:10,accuracy);
title('K nearest neighborhood classifier');
xlabel('K') 
ylabel('Accuracy') 
savepath_str = sprintf("D:/homework/nctu/senior/fall/machine_learning/hw/hw2/result_1/3/accuracy.png");
savepath_chr=convertStringsToChars(savepath_str);
saveas(gcf,savepath_chr);

%------------------------------------------------------------------------------------------------------------
%PCA
N=120;
pca_x_norm=normalize(x(1:120,:),'zscore');
pca_x_norm_t=normalize(x(121:158,:),'zscore');
pca_mean=(1/N)*(sum(pca_x_norm));
pca_x=pca_x_norm-pca_mean;
pca_s=(1/(N-1))*(pca_x_norm'*pca_x_norm);
[pca_V,pca_D]=eig(pca_s);
[d,ind] = sort(diag(pca_D),'descend');
Ds = pca_D(ind,ind);
Vs = pca_V(:,ind);

for dim =1:9
    %dim=9;%PCA should be used to reduce the dimension of face images to 2, 5 and 10
    pca_filter=Vs(:,1:dim);
    pca=pca_filter'*pca_x_norm';
    pca_t=pca_filter'*pca_x_norm_t';
    pca_x_train=pca';
    pca_x_test=pca_t';
    pca_y_train=y(1:120,:);
    pca_y_test=y(121:158,:);

    pca_dis=zeros(120,38);
    pca_sort_index_result=zeros(120,38);
    pca_dis_sort_result=zeros(120,38);
    pca_t_train=zeros(120,3,38);
    for j=1:38
        pca_r_x_test = repmat(pca_x_test(j,:),120,1) ;
        pca_V = pca_r_x_test - pca_x_train;
        pca_D = pca_V.^(2);
        pca_dis(:,j)=sqrt(sum(pca_D,2));%each column is about the point to the others
        [pca_dis_sort,pca_sort_index] = sort(pca_dis(:,j),'ascend');
        pca_sort_index_result(:,j)=pca_sort_index;
        pca_dis_sort_result(:,j)=pca_dis_sort;
        pca_t_train(:,:,j)=pca_y_train(pca_sort_index,:); 
    end
    pca_result=zeros(38,10);
    pca_accuracy=zeros(1,10);
    for K=1:10
        for j=1:38
            if K==1
                pca_temp=pca_t_train(1:K,:,j);
            else
                pca_temp=sum(pca_t_train(1:K,:,j));
            end
            [pca_M,pca_class] = max(pca_temp);
            pca_result(j,K)=pca_class;
        end
        [pca_tempy,pca_y_test_class]=max(pca_y_test,[],2);
        pca_accuracy_matrix = pca_result(:,K)==pca_y_test_class;
        pca_accuracy(:,K)=sum(pca_accuracy_matrix)/38.0;
    end
    figure
    plot(1:10,pca_accuracy);
    title_str = sprintf("K nearest neighborhood classifier with PCA(dimension="+int2str(dim)+")");
    title_chr=convertStringsToChars(title_str);
    title(title_chr);
    xlabel('K') 
    ylabel('Accuracy') 
    savepath_str = sprintf("D:/homework/nctu/senior/fall/machine_learning/hw/hw2/result_1/3/accuracy_pca_"+int2str(dim)+".png");
    savepath_chr=convertStringsToChars(savepath_str);
    saveas(gcf,savepath_chr);
end