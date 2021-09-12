%readfile
filename = 'dataset/1_data.mat';
data =cell2mat(struct2cell(load(filename)));
x=data(1:100,:);
x_sort=sort(x);
t=data(101:200,:);
%set parameters
M=3;
N=5;
s=0.1;
alpha=10^(-6);
mean_prior=[0;0;0];
variance_inv_prior=alpha*eye(3);
beta=1;
%set mu_j
mu0=0;
mu1=(2*1)/M;
mu2=(2*2)/M;
phi=[sigmoid((x-mu0)./s),sigmoid((x-mu1)./s),sigmoid((x-mu2)./s)];
for times=1:4
    for i=1:N
        phi_temp=phi(1:i,:);
        variance_inv=variance_inv_prior+beta*(phi_temp'*phi_temp);
        mean=pinv(variance_inv)*(variance_inv_prior*mean_prior+beta*phi_temp'*t(1:i,:));
        variance_inv_prior= variance_inv;
        mean_prior=mean;
    end
    w = mvnrnd(mean,pinv(variance_inv),1);
    plot_x_init=linspace(0,2,100)';
    plot_x=[sigmoid((plot_x_init-mu0)./s),sigmoid((plot_x_init-mu1)./s),sigmoid((plot_x_init-mu2)./s)];
    y_norm=plot_x*w';
    mean=plot_x*mean;
    variance=zeros(100,1);
    for sample=1:100
        variance(sample,:)=(1/beta)+plot_x(sample,:)*pinv(variance_inv)*plot_x(sample,:)';
    end    
    
    %plot fig
    figure
    legend
    hold on
    plot(plot_x_init,mean,'DisplayName','mean');
%     plot_temp=[x, w',mean,variance];
%     plot_result = sortrows(plot_temp);
%     plot(plot_result(:,1),plot_result(:,2),'DisplayName','prediction');
%     plot(plot_result(:,1),plot_result(:,3),'DisplayName','mean');
    y1 = mean+variance;
    y2 = mean-variance;
    opts={'EdgeColor','none','FaceColor', [0.9290 0.6940 0.1250],'FaceAlpha',0.5,'DisplayName','std'};
    fill_between(plot_x_init, y1, y2, y1>y2, opts{:});
%     y_norm=zeros(100,1);
%     for z=1:100
%         y_norm(z,1) = max(normpdf(plot_x_init,mean(z,1),variance(z,1)));
%     end
    plot(plot_x_init,y_norm,'DisplayName','prediction');
    hold off
    
    savepath_str = sprintf("D:/homework/nctu/senior/fall/machine_learning/hw/hw2/result_1/1/q2_"+"%d.png", times);
    savepath_chr=convertStringsToChars(savepath_str);
    saveas(gcf,savepath_chr);
    %set N value
    if times==1
        N=10;
    elseif times==2
        N=30;
    else
        N=80; 
    end
end

