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
x12=x_pca(1:200,:);
x23=x_pca(101:300,:);
x13=x_pca([1:100,201:300],:);
t12=t_norm(1:200,1);
t23=t_norm(101:300,2);
t13=t_norm([1:100,201:300],1);

[y_lin12,w_lin12,bias_lin12,sv12]=svm_linear(x12, t12', C, tol);
[y_lin23,w_lin23,bias_lin23,sv23]=svm_linear(x23, t23', C, tol);
[y_lin13,w_lin13,bias_lin13,sv13]=svm_linear(x13, t13', C, tol);
y_lin12_new=[y_lin12;zeros(100,1)];
y_lin23_new=[zeros(100,1);y_lin23];
y_lin13_new=[y_lin13(1:100);zeros(100,1);y_lin13(101:200,:)];

% y_index=zeros(300,1);
y_index1=[find(y_lin12_new>0 & y_lin13_new>0);(find(y_lin12_new>0 & y_lin13_new<0 & abs(y_lin12_new)>abs(y_lin13_new)));(find(y_lin12_new<0 & y_lin13_new>0 & abs(y_lin12_new)<abs(y_lin13_new)));(find(y_lin12_new>0 & y_lin23_new>0 & abs(y_lin12_new)>abs(y_lin23_new)));(find(y_lin13_new>0 & y_lin23_new<0 & abs(y_lin13_new)>abs(y_lin23_new)))];
y_index2=[find(y_lin12_new<0 & y_lin23_new>0);(find(y_lin12_new<0 & y_lin23_new<0 & abs(y_lin12_new)>abs(y_lin23_new)));(find(y_lin12_new>0 & y_lin23_new>0 & abs(y_lin12_new)<abs(y_lin23_new)));(find(y_lin12_new<0 & y_lin13_new>0 & abs(y_lin12_new)>abs(y_lin13_new)));(find(y_lin13_new<0 & y_lin23_new>0 & abs(y_lin13_new)<abs(y_lin23_new)))];
y_index3=[find(y_lin13_new<0 & y_lin23_new<0);(find(y_lin13_new<0 & y_lin23_new>0 & abs(y_lin13_new)>abs(y_lin23_new)));(find(y_lin13_new>0 & y_lin23_new<0 & abs(y_lin13_new)<abs(y_lin23_new)));(find(y_lin12_new>0 & y_lin13_new<0 & abs(y_lin12_new)<abs(y_lin13_new)));(find(y_lin12_new<0 & y_lin23_new<0 & abs(y_lin12_new)<abs(y_lin23_new)))];
y=zeros(300,1);
y(y_index1)=0;
y(y_index2)=1;
y(y_index3)=2;
rms=sqrt(mean((y-t).^2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
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

y_lin12_draw=create_phi*w_lin12'+bias_lin12;
y_lin23_draw=create_phi*w_lin23'+bias_lin23;
y_lin13_draw=create_phi*w_lin13'+bias_lin13;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_index1_draw=[find(y_lin12_draw>0 & y_lin13_draw>0);(find(y_lin12_draw>0 & y_lin13_draw<0 & abs(y_lin12_draw)>abs(y_lin13_draw)));(find(y_lin12_draw<0 & y_lin13_draw>0 & abs(y_lin12_draw)<abs(y_lin13_draw)));(find(y_lin12_draw>0 & y_lin23_draw>0 & abs(y_lin12_draw)>abs(y_lin23_draw)));(find(y_lin13_draw>0 & y_lin23_draw<0 & abs(y_lin13_draw)>abs(y_lin23_draw)))];
y_index2_draw=[find(y_lin12_draw<0 & y_lin23_draw>0);(find(y_lin12_draw<0 & y_lin23_draw<0 & abs(y_lin12_draw)>abs(y_lin23_draw)));(find(y_lin12_draw>0 & y_lin23_draw>0 & abs(y_lin12_draw)<abs(y_lin23_draw)));(find(y_lin12_draw<0 & y_lin13_draw>0 & abs(y_lin12_draw)>abs(y_lin13_draw)));(find(y_lin13_draw<0 & y_lin23_draw>0 & abs(y_lin13_draw)<abs(y_lin23_draw)))];
y_index3_draw=[find(y_lin13_draw<0 & y_lin23_draw<0);(find(y_lin13_draw<0 & y_lin23_draw>0 & abs(y_lin13_draw)>abs(y_lin23_draw)));(find(y_lin13_draw>0 & y_lin23_draw<0 & abs(y_lin13_draw)<abs(y_lin23_draw)));(find(y_lin12_draw>0 & y_lin13_draw<0 & abs(y_lin12_draw)<abs(y_lin13_draw)));(find(y_lin12_draw<0 & y_lin23_draw<0 & abs(y_lin12_draw)<abs(y_lin23_draw)))];
y_draw=zeros(40000,1);
y_draw(y_index1_draw)=0;
y_draw(y_index2_draw)=1;
y_draw(y_index3_draw)=2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot
figure;
hold on
title('Linear Kernel one v.s. one')
plot(create_phi(y_draw==0,1),create_phi(y_draw==0,2),'.','Color',[1 0.7 0.7]); 
plot(create_phi(y_draw==1,1),create_phi(y_draw==1,2),'.','Color',[0.7 1 0.7]); 
plot(create_phi(y_draw==2,1),create_phi(y_draw==2,2),'.','Color',[0.7 0.7 1]); 
plot(sv12(:,1),sv12(:,2),'o','Color', [0.17 0.17 0.17]); 
plot(sv23(:,1),sv23(:,2),'o','Color', [0.17 0.17 0.17]); 
s=plot(sv13(:,1),sv13(:,2),'o','Color', [0.17 0.17 0.17]);
h1=plot(x_pca(1:100,1),x_pca(1:100,2),'rx'); 
h2=plot(x_pca(101:200,1),x_pca(101:200,2),'+','Color',[0 0.4 0]); 
h3=plot(x_pca(201:300,1),x_pca(201:300,2),'b*'); 
legend([h1 h2 h3 s],{'T-shirt','Trouser','Pullover','support vector'})
savepath_str = sprintf("./q2_lin_one_one.png");
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