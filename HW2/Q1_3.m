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
    R = mvnrnd(mean,pinv(variance_inv),100000);
    figure
    plot(R(:,1),R(:,2),'+');
    savepath_str = sprintf("D:/homework/nctu/senior/fall/machine_learning/hw/hw2/result_1/1/q3_1_"+"%d.png", times);
    savepath_chr=convertStringsToChars(savepath_str);
    saveas(gcf,savepath_chr);

    %draw
    Xmin=min(R(:,1));
    Xmax=max(R(:,1));
    Ymin=min(R(:,2));
    Ymax=max(R(:,2));
    %split size
    Nx=100;
    Ny=100;
    Xedge=linspace(Xmin,Xmax,Nx);
    Yedge=linspace(Ymin,Ymax,Ny);
    %count point per area
    [N,~,~,binX,binY] = histcounts2(R(:,1),R(:,2),[-inf,Xedge(2:end-1),inf],[-inf,Yedge(2:end-1),inf]);
    XedgeM=movsum(Xedge,2)/2;
    YedgeM=movsum(Yedge,2)/2;
    [Xedgemesh,Yedgemesh]=meshgrid(XedgeM(2:end),YedgeM(2:end));
    % draw pcolor
    figure
    pcolor(Xedgemesh,Yedgemesh,N');shading flat
    
    savepath_str = sprintf("D:/homework/nctu/senior/fall/machine_learning/hw/hw2/result_1/1/q3_2_"+"%d.png", times);
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

