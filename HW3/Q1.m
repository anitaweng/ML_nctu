%readfile
filename = 'gp.mat';
data =cell2mat(struct2cell(load(filename)));
x=data(1:100,:);
t=data(101:200,:);
x_train=x(1:60,:);
x_test=x(1:40,:);
t_train=t(1:60,:);
t_test=t(1:40,:);

inv_beta=1;
theta=[1,10,1,1];%set theta combination

diff=zeros(size(x_train,1),size(x_test,1));
delta=eye(size(x_train,1));
k=zeros(size(x_train,1),size(x_train,1));
for i=1:size(x_train,1)
    k(:,i)=kernel(x_train,x_train(i,1),theta);
end
c_n=k+inv_beta*delta;

%train set
k_vet_train=zeros(size(x_train,1),size(x_train,1));
c_draw_train=zeros(size(x_train,1),1);
m_train=zeros(size(x_train,1),1);
var_train=zeros(size(x_train,1),1);
for N=1:size(x_train,1)
    k_vet_train(:,N)=kernel(x_train,x_train(N,1),theta);
    c_draw_train(N,1)=kernel(x_train(N,1),x_train(N,1),theta)+inv_beta;
    m_train(N,1)=k_vet_train(:,N)'*pinv(c_n)*t_train;
    var_train(N,1)=c_draw_train(N,1)-k_vet_train(:,N)'*pinv(c_n)*k_vet_train(:,N);
end
%rms error
error_train=sqrt(mean((m_train-t_train).^2));

%draw
draw_size=100;
x_draw=linspace(0,2,draw_size)';
k_vet=zeros(size(x_train,1),draw_size);
c_draw=zeros(draw_size,1);
m=zeros(draw_size,1);
var=zeros(draw_size,1);
for N=1:draw_size
    k_vet(:,N)=kernel(x_train,x_draw(N,1),theta);
    c_draw(N,1)=kernel(x_draw(N,1),x_draw(N,1),theta)+inv_beta;
    m(N,1)=k_vet(:,N)'*pinv(c_n)*t_train;
    var(N,1)=c_draw(N,1)-k_vet(:,N)'*pinv(c_n)*k_vet(:,N);
end

%test set
k_vet_pred=zeros(size(x_train,1),size(x_test,1));
c_draw_pred=zeros(size(x_test,1),1);
m_pred=zeros(size(x_test,1),1);
var_pred=zeros(size(x_test,1),1);
for N=1:size(x_test,1)
    k_vet_pred(:,N)=kernel(x_train,x_test(N,1),theta);
    c_draw_pred(N,1)=kernel(x_test(N,1),x_test(N,1),theta)+inv_beta;
    m_pred(N,1)=k_vet_pred(:,N)'*pinv(c_n)*t_train;
    var_pred(N,1)=c_draw_pred(N,1)-k_vet_pred(:,N)'*pinv(c_n)*k_vet_pred(:,N);
end
%rms error
error_test=sqrt(mean((m_pred-t_test).^2));

figure
legend
hold on
title(['Gaussian Process Example \theta = [',num2str(theta(1)),',',num2str(theta(2)),',',num2str(theta(3)),',',num2str(theta(4)),']'])
plot(x_draw,m,'DisplayName','mean');
plot(x_test,m_pred,'.','DisplayName','pred')
plot(x_train,t_train,'o','DisplayName','test','color','g')
y1 = m+sqrt(var);
y2 = m-sqrt(var);
opts={'EdgeColor','none','FaceColor', [0.9290 0.6940 0.1250],'FaceAlpha',0.5,'DisplayName','std'};
fill_between(x_draw, y1, y2, y1>y2, opts{:});
saveas(gcf,['./q1_without_ard_',num2str(theta(1)),'_',num2str(theta(2)),'_',num2str(theta(3)),'_',num2str(theta(4)),'.png']);
hold off


%def kernel function
function k = kernel(xn,xm,theta)
    diff=xn-xm;
    k=theta(1)*exp(-theta(2)*0.5*diff.^2)+theta(3)+theta(4)*(xn*xm);
end
