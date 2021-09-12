%readfile
filename = 'gp.mat';
data =cell2mat(struct2cell(load(filename)));
x=data(1:100,:);
t=data(101:200,:);
% t=normalize(t,'zscore');
x_train=x(1:60,:);
x_test=x(1:40,:);
t_train=t(1:60,:);
t_test=t(1:40,:);
% t_train=normalize(t_train,'zscore');

inv_beta=1;
theta=[1,10,0,0];%set theta combination 3 3 3 3;2 2 2 2;4 4 4 4;10 10 10 10;5 5 10 10;1 2 0 0

diff=zeros(size(x_train,1),size(x_test,1));
delta=eye(size(x_train,1));
k=zeros(size(x_train,1),size(x_train,1));
for i=1:size(x_train,1)
    k(:,i)=kernel(x_train,x_train(i,1),theta);
end
c_n=k+inv_beta*delta;
x1=repmat(x_train,1,60);
x2=repmat(x_train',60,1);
theta = ARD(x1,x2,theta,t_train,inv_beta,delta);
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
saveas(gcf,['./q1_ard_',num2str(theta(1)),'_',num2str(theta(2)),'_',num2str(theta(3)),'_',num2str(theta(4)),'.png']);
hold off

%def kernel function
function k = kernel(xn,xm,theta)
    diff=xn-xm;
    k=theta(1)*exp(-theta(2)*0.5*diff.^2)+theta(3)+theta(4)*(xn*xm);
end

function theta_new = ARD(xn,xm,theta,t,inv_beta,delta)
    theta_old=theta;
    eta=0.0001;
    num=0;
    while 1
        num=num+1;
        k=zeros(size(xn(:,1),1),size(xn(:,1),1));
        for i=1:size(xn(:,1),1)
            k(:,i)=kernel(xn(:,1),xn(i,1),theta_old);
        end
        c_n=k+inv_beta*delta;
        diff=xn-xm;
        d_theta0=exp(-theta_old(2)*0.5*diff.^2);
        d_theta1=theta_old(1)*(-0.5*diff.^2)*exp(-theta_old(2)*0.5*diff.^2);
        d_theta2=ones(size(xn));
        d_theta3=(xn(:,1)*xn(:,1)');
        theta_new=zeros(1,4);
        theta_new(1)=theta_old(1)+eta*(-0.5*trace(pinv(c_n)*d_theta0)+0.5*t'*pinv(c_n)*d_theta0*pinv(c_n)*t);
        theta_new(2)=theta_old(2)+eta*(-0.5*trace(pinv(c_n)*d_theta1)+0.5*t'*pinv(c_n)*d_theta1*pinv(c_n)*t);
        theta_new(3)=theta_old(3)+eta*(-0.5*trace(pinv(c_n)*d_theta2)+0.5*t'*pinv(c_n)*d_theta2*pinv(c_n)*t);
        theta_new(4)=theta_old(4)+eta*(-0.5*trace(pinv(c_n)*d_theta3)+0.5*t'*pinv(c_n)*d_theta3*pinv(c_n)*t);
        if max((theta_new-theta_old))<0.9
            break
        end
        theta_old=theta_new;
    end
end

