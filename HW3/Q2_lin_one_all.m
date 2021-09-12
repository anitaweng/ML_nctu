%readfile
filename = 'x_train.csv';
x = xlsread(filename);
filename = 't_train.csv';
t = xlsread(filename);
t_norm=zeros(size(t,1),3);
for i =1:size(t,1) % number of pic
    if t(i)==0
        t_norm(i,:)=[1,-1,-1];
    elseif t(i)==1
        t_norm(i,:)=[-1,1,-1];
    else
        t_norm(i,:)=[-1,-1,1];
    end    
end
C=1000;
tol=0.001;
%PCA
N=300;
X=x;
pca_mean=(1/N)*(sum(X));
pca_x=X-pca_mean;
pca_s=(1/(N-1))*(X'*X);
[V,D]=eig(pca_s);
[d,ind] = sort(diag(D),'descend');
Ds = D(ind,ind);
Vs = V(:,ind);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dim=2;%PCA should be used to reduce the dimension of image to 2
filter=Vs(:,1:dim);

x_pca=x*filter;
% x_pca=normalize(x_pca,'zscore');
x_pca=x_pca/(max(max(x_pca))-min(min(x_pca)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[y_lin(:,1),w_lin(1,:),bias_lin(1),sv1]=svm_linear(x_pca, t_norm(:,1)', C, tol);
[y_lin(:,2),w_lin(2,:),bias_lin(2),sv2]=svm_linear(x_pca, t_norm(:,2)', C, tol);
[y_lin(:,3),w_lin(3,:),bias_lin(3),sv3]=svm_linear(x_pca, t_norm(:,3)', C, tol);

[y,index]=max(y_lin,[],2);
index=index-1;
rms=sqrt(mean((index-t).^2));

num=0;
draw_size=200;
xaxis=linspace(0,1,draw_size);
yaxis=linspace(-0.5,0.5,draw_size);
create_phi=zeros(draw_size^2,dim);
for i = 1:draw_size
    for j = 1:draw_size
       num=num+1;
       create_phi(num,:) = [xaxis(i),yaxis(j)]; 
    end
end

y_draw(:,1)=create_phi*w_lin(1,:)'+bias_lin(1);
y_draw(:,2)=create_phi*w_lin(2,:)'+bias_lin(2);
y_draw(:,3)=create_phi*w_lin(3,:)'+bias_lin(3);
[y_draw_index,index_draw]=max(y_draw,[],2);
index_draw=index_draw-1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot
figure
hold on
title('Linear Kernel one v.s. all')
plot(create_phi(index_draw==0,1),create_phi(index_draw==0,2),'.','Color',[1 0.7 0.7]); 
plot(create_phi(index_draw==1,1),create_phi(index_draw==1,2),'.','Color',[0.7 1 0.7]);
plot(create_phi(index_draw==2,1),create_phi(index_draw==2,2),'.','Color',[0.7 0.7 1]); 
plot(sv1(:,1),sv1(:,2),'o','Color', [0.17 0.17 0.17]); 
plot(sv2(:,1),sv2(:,2),'o','Color', [0.17 0.17 0.17]); 
s=plot(sv3(:,1),sv3(:,2),'o','Color', [0.17 0.17 0.17]);
h1=plot(x_pca(1:100,1),x_pca(1:100,2),'rx'); 
h2=plot(x_pca(101:200,1),x_pca(101:200,2),'+','Color',[0 0.4 0]); 
h3=plot(x_pca(201:300,1),x_pca(201:300,2),'b*'); 
legend([h1 h2 h3 s],{'T-shirt','Trouser','Pullover','support vector'})
savepath_str = sprintf("./q2_lin_one_all.png");
savepath_chr=convertStringsToChars(savepath_str);
saveas(gcf,savepath_chr);
hold off

function [y,w,bias,sv]=svm_linear(x, t, C, tol)
    phi=x;
    kernel=phi*phi';
    [alpha,bias]=smo(kernel, t, C, tol);
    w=(alpha.*t)*phi;
    y=phi*w'+bias;
    sv_index=alpha>0;
    sv=x(sv_index,:);
end