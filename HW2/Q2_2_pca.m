%read image
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

%PCA
N=50;
X=[input1(1:10304,:),input2(1:10304,:),input3(1:10304,:),input4(1:10304,:),input5(1:10304,:)]';
pca_mean=(1/N)*(sum(X));
pca_x=X-pca_mean;
pca_s=(1/(N-1))*(X'*X);
[V,D]=eig(pca_s);
[d,ind] = sort(diag(D),'descend');
Ds = D(ind,ind);
Vs = V(:,ind);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%run Q2_2.m