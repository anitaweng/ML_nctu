%readfile
filename = 'Dataset/dataset_X.csv';
x = xlsread(filename);
filename = 'Dataset/dataset_T.csv';
y = xlsread(filename);
%--------------------------- M=1-------------------------------------------
% x_train1=x(1:822,:);
% x_val1=x(823:1096,:);
% y_train1=y(1:822,:);
% y_val1=y(823:1096,:);
% 
% x_train2=x([1:548,823:1096],:);
% x_val2=x(549:822,:);
% y_train2=y([1:548,823:1096],:);
% y_val2=y(549:822,:);
% 
% x_train3=x([1:274,549:1096],:);
% x_val3=x(275:548,:);
% y_train3=y([1:274,549:1096],:);
% y_val3=y(275:548,:);
% 
% x_train4=x(275:1096,:);
% x_val4=x(1:274,:);
% y_train4=y(275:1096,:);
% y_val4=y(1:274,:);

%--------------------------M=2---------------------------------------------
x_train1=x(1:822,:);
x_val1=x(823:1096,:);
% x_train_trim=x(1:822,[3,8,9,11,14]);
% x_val_trim=x(823:1096,[3,8,9,11,14]);
x_train_trim=x(1:822,:);
x_val_trim=x(823:1096,:);

% phi_temp=zeros(822,32);
% order2=zeros(1,15);
phi_temp=zeros(822,170);
order2=zeros(1,152);

for i=1:size(x_train_trim,1)
    n=0;
    for j = 1:size(x_train_trim,2)
        for k = j:size(x_train_trim,2)
            n=n+1;
            order2(n)=x_train_trim(i,j)*x_train_trim(i,k);
        end
    end
   phi_temp(i,:)=[x_train1(i,:),order2];
end

% phi2_test_temp=zeros(274,32);
% order2=zeros(1,15);
phi2_test_temp=zeros(274,170);
order2=zeros(1,152);

for i=1:size(x_val_trim,1)
    n=0;
    for j = 1:size(x_val_trim,2)
        for k = j:size(x_val_trim,2)
            n=n+1;
            order2(n)=x_val_trim(i,j)*x_val_trim(i,k);
        end
    end
    phi2_test_temp(i,:)=[x_val1(i,:),order2];
end
x_train1=phi_temp;
x_val1=phi2_test_temp;
y_train1=y(1:822,:);
y_val1=y(823:1096,:);
%--------------------------train 1 end ------------------------------------
x_train2=x([1:548,823:1096],:);
x_val2=x(549:822,:);

% x_train_trim=x([1:548,823:1096],[3,8,9,11,14]);
% x_val_trim=x(549:822,[3,8,9,11,14]);
x_train_trim=x([1:548,823:1096],:);
x_val_trim=x(549:822,:);

% phi_temp=zeros(822,32);
% order2=zeros(1,15);
phi_temp=zeros(822,170);
order2=zeros(1,152);

for i=1:size(x_train_trim,1)
    n=0;
    for j = 1:size(x_train_trim,2)
        for k = j:size(x_train_trim,2)
            n=n+1;
            order2(n)=x_train_trim(i,j)*x_train_trim(i,k);
        end
    end
   phi_temp(i,:)=[x_train2(i,:),order2];
end

% phi2_test_temp=zeros(274,32);
% order2=zeros(1,15);
phi2_test_temp=zeros(274,170);
order2=zeros(1,152);

for i=1:size(x_val_trim,1)
    n=0;
    for j = 1:size(x_val_trim,2)
        for k = j:size(x_val_trim,2)
            n=n+1;
            order2(n)=x_val_trim(i,j)*x_val_trim(i,k);
        end
    end
    phi2_test_temp(i,:)=[x_val2(i,:),order2];
end
x_train2=phi_temp;
x_val2=phi2_test_temp;
y_train2=y([1:548,823:1096],:);
y_val2=y(549:822,:);
%--------------------------train 2 end ------------------------------------
x_train3=x([1:274,549:1096],:);
x_val3=x(275:548,:);

% x_train_trim=x([1:274,549:1096],[3,8,9,11,14]);
% x_val_trim=x(275:548,[3,8,9,11,14]);
x_train_trim=x([1:274,549:1096],:);
x_val_trim=x(275:548,:);

% phi_temp=zeros(822,32);
% order2=zeros(1,15);
phi_temp=zeros(822,170);
order2=zeros(1,152);

for i=1:size(x_train_trim,1)
    n=0;
    for j = 1:size(x_train_trim,2)
        for k = j:size(x_train_trim,2)
            n=n+1;
            order2(n)=x_train_trim(i,j)*x_train_trim(i,k);
        end
    end
   phi_temp(i,:)=[x_train3(i,:),order2];
end

% phi2_test_temp=zeros(274,32);
% order2=zeros(1,15);
phi2_test_temp=zeros(274,170);
order2=zeros(1,152);

for i=1:size(x_val_trim,1)
    n=0;
    for j = 1:size(x_val_trim,2)
        for k = j:size(x_val_trim,2)
            n=n+1;
            order2(n)=x_val_trim(i,j)*x_val_trim(i,k);
        end
    end
    phi2_test_temp(i,:)=[x_val3(i,:),order2];
end
x_train3=phi_temp;
x_val3=phi2_test_temp;
y_train3=y([1:274,549:1096],:);
y_val3=y(275:548,:);
%--------------------------train 3 end ------------------------------------
x_train4=x(275:1096,:);
x_val4=x(1:274,:);

% x_train_trim=x(275:1096,[3,8,9,11,14]);
% x_val_trim=x(1:274,[3,8,9,11,14]);
x_train_trim=x(275:1096,:);
x_val_trim=x(1:274,:);

% phi_temp=zeros(822,32);
% order2=zeros(1,15);
phi_temp=zeros(822,170);
order2=zeros(1,152);

for i=1:size(x_train_trim,1)
    n=0;
    for j = 1:size(x_train_trim,2)
        for k = j:size(x_train_trim,2)
            n=n+1;
            order2(n)=x_train_trim(i,j)*x_train_trim(i,k);
        end
    end
    phi_temp(i,:)=[x_train4(i,:),order2];
end

% phi2_test_temp=zeros(274,32);
% order2=zeros(1,15);
phi2_test_temp=zeros(274,170);
order2=zeros(1,152);

for i=1:size(x_val_trim,1)
    n=0;
    for j = 1:size(x_val_trim,2)
        for k = j:size(x_val_trim,2)
            n=n+1;
            order2(n)=x_val_trim(i,j)*x_val_trim(i,k);
        end
    end
    phi2_test_temp(i,:)=[x_val4(i,:),order2];
end
x_train4=phi_temp;
x_val4=phi2_test_temp;
y_train4=y(275:1096,:);
y_val4=y(1:274,:);

%normalize data 1----------------------------------------------------------
% x_train1=(x_train1-mean(x_train1)).*(1./std(x_train1));
% y_train1=(y_train1-mean(y_train1)).*(1./std(y_train1));
% x_val1=(x_val1-mean(x_val1)).*(1./std(x_val1));
% y_val1=(y_val1-mean(y_val1)).*(1./std(y_val1));

mu1=mean(x_train1);
mu_test1=mean(x_val1);
sigma1=std(x_train1);
sigma_test1=std(x_val1);
phi_poly1=[ones(size(x_train1,1),1),x_train1];
phi_gauss1=exp(-(x_train1-mu1).^2./(2*sigma1.^2));
phi_gauss1=[ones(size(x_train1,1),1),phi_gauss1];
phi_sig1=1./(1.+exp(-(x_train1-mu1)./sigma1));
phi_sig1=[ones(size(x_train1,1),1),phi_sig1];
phi_test_gauss1=exp(-(x_val1-mu_test1).^2./(2*sigma_test1.^2));
phi_test_gauss1=[ones(size(x_val1,1),1),phi_test_gauss1];
phi_test_sig1=1./(1.+exp(-(x_val1-mu_test1)./sigma_test1));
phi_test_sig1=[ones(size(x_val1,1),1),phi_test_sig1];
phi_test_poly1=[ones(size(x_val1,1),1),x_val1];
w_poly1=pinv(phi_poly1'*phi_poly1)*phi_poly1'*y_train1;
w_gauss1=pinv(phi_gauss1'*phi_gauss1)*phi_gauss1'*y_train1;
w_sig1=pinv(phi_sig1'*phi_sig1)*phi_sig1'*y_train1;
%error=0.5*sqrt(sum((y_train-phi*w).^2));
error_gauss1=sqrt(mean((phi_gauss1*w_gauss1-y_train1).^2));
error_sig1=sqrt(mean((phi_sig1*w_sig1-y_train1).^2));
error_poly1=sqrt(mean((phi_poly1*w_poly1-y_train1).^2));
%error_test=0.5*sqrt(sum((y_val-phi_test*w).^2));
error_test_gauss1=sqrt(mean((phi_test_gauss1*w_gauss1-y_val1).^2));
error_test_sig1=sqrt(mean((phi_test_sig1*w_sig1-y_val1).^2));
error_test_poly1=sqrt(mean((phi_test_poly1*w_poly1-y_val1).^2));

%normalize data 2----------------------------------------------------------
% x_train2=(x_train2-mean(x_train2)).*(1./std(x_train2));
% y_train2=(y_train2-mean(y_train2)).*(1./std(y_train2));
% x_val2=(x_val2-mean(x_val2)).*(1./std(x_val2));
% y_val2=(y_val2-mean(y_val2)).*(1./std(y_val2));

mu2=mean(x_train2);
mu_test2=mean(x_val2);
sigma2=std(x_train2);
sigma_test2=std(x_val2);
phi_poly2=[ones(size(x_train2,1),1),x_train2];
phi_gauss2=exp(-(x_train2-mu2).^2./(2*sigma2.^2));
phi_gauss2=[ones(size(x_train2,1),1),phi_gauss2];
phi_sig2=1./(1.+exp(-(x_train2-mu2)./sigma2));
phi_sig2=[ones(size(x_train2,1),1),phi_sig2];
phi_test_gauss2=exp(-(x_val2-mu_test2).^2./(2*sigma_test2.^2));
phi_test_gauss2=[ones(size(x_val2,1),1),phi_test_gauss2];
phi_test_sig2=1./(1.+exp(-(x_val2-mu_test2)./sigma_test2));
phi_test_sig2=[ones(size(x_val2,1),1),phi_test_sig2];
phi_test_poly2=[ones(size(x_val2,1),1),x_val2];
w_poly2=pinv(phi_poly2'*phi_poly2)*phi_poly2'*y_train2;
w_gauss2=pinv(phi_gauss2'*phi_gauss2)*phi_gauss2'*y_train2;
w_sig2=pinv(phi_sig2'*phi_sig2)*phi_sig2'*y_train2;
%error=0.5*sqrt(sum((y_train-phi*w).^2));
error_gauss2=sqrt(mean((phi_gauss2*w_gauss2-y_train2).^2));
error_sig2=sqrt(mean((phi_sig2*w_sig2-y_train2).^2));
error_poly2=sqrt(mean((phi_poly2*w_poly2-y_train2).^2));
%error_test=0.5*sqrt(sum((y_val-phi_test*w).^2));
error_test_gauss2=sqrt(mean((phi_test_gauss2*w_gauss2-y_val2).^2));
error_test_sig2=sqrt(mean((phi_test_sig2*w_sig2-y_val2).^2));
error_test_poly2=sqrt(mean((phi_test_poly2*w_poly2-y_val2).^2));

%normalize data 3----------------------------------------------------------
% x_train3=(x_train3-mean(x_train3)).*(1./std(x_train3));
% y_train3=(y_train3-mean(y_train3)).*(1./std(y_train3));
% x_val3=(x_val3-mean(x_val3)).*(1./std(x_val3));
% y_val3=(y_val3-mean(y_val3)).*(1./std(y_val3));

mu3=mean(x_train3);
mu_test3=mean(x_val3);
sigma3=std(x_train3);
sigma_test3=std(x_val3);
phi_poly3=[ones(size(x_train3,1),1),x_train3];
phi_gauss3=exp(-(x_train3-mu3).^2./(2*sigma3.^2));
phi_gauss3=[ones(size(x_train3,1),1),phi_gauss3];
phi_sig3=1./(1.+exp(-(x_train3-mu3)./sigma3));
phi_sig3=[ones(size(x_train3,1),1),phi_sig3];
phi_test_gauss3=exp(-(x_val3-mu_test3).^2./(2*sigma_test3.^2));
phi_test_gauss3=[ones(size(x_val3,1),1),phi_test_gauss3];
phi_test_sig3=1./(1.+exp(-(x_val3-mu_test3)./sigma_test3));
phi_test_sig3=[ones(size(x_val3,1),1),phi_test_sig3];
phi_test_poly3=[ones(size(x_val3,1),1),x_val3];
w_poly3=pinv(phi_poly3'*phi_poly3)*phi_poly3'*y_train3;
w_gauss3=pinv(phi_gauss3'*phi_gauss3)*phi_gauss3'*y_train3;
w_sig3=pinv(phi_sig3'*phi_sig3)*phi_sig3'*y_train3;
%error=0.5*sqrt(sum((y_train-phi*w).^2));
error_gauss3=sqrt(mean((phi_gauss3*w_gauss3-y_train3).^2));
error_sig3=sqrt(mean((phi_sig3*w_sig3-y_train3).^2));
error_poly3=sqrt(mean((phi_poly3*w_poly3-y_train3).^2));
%error_test=0.5*sqrt(sum((y_val-phi_test*w).^2));
error_test_gauss3=sqrt(mean((phi_test_gauss3*w_gauss3-y_val3).^2));
error_test_sig3=sqrt(mean((phi_test_sig3*w_sig3-y_val3).^2));
error_test_poly3=sqrt(mean((phi_test_poly3*w_poly3-y_val3).^2));

%normalize data 4----------------------------------------------------------
% x_train4=(x_train4-mean(x_train4)).*(1./std(x_train4));
% y_train4=(y_train4-mean(y_train4)).*(1./std(y_train4));
% x_val4=(x_val4-mean(x_val4)).*(1./std(x_val4));
% y_val4=(y_val4-mean(y_val4)).*(1./std(y_val4));

mu4=mean(x_train4);
mu_test4=mean(x_val4);
sigma4=std(x_train4);
sigma_test4=std(x_val4);
phi_poly4=[ones(size(x_train4,1),1),x_train4];
phi_gauss4=exp(-(x_train4-mu4).^2./(2*sigma4.^2));
phi_gauss4=[ones(size(x_train4,1),1),phi_gauss4];
phi_sig4=1./(1.+exp(-(x_train4-mu4)./sigma4));
phi_sig4=[ones(size(x_train4,1),1),phi_sig4];
phi_test_gauss4=exp(-(x_val4-mu_test4).^2./(2*sigma_test4.^2));
phi_test_gauss4=[ones(size(x_val4,1),1),phi_test_gauss4];
phi_test_sig4=1./(1.+exp(-(x_val4-mu_test4)./sigma_test4));
phi_test_sig4=[ones(size(x_val4,1),1),phi_test_sig4];
phi_test_poly4=[ones(size(x_val4,1),1),x_val4];
w_poly4=pinv(phi_poly4'*phi_poly4)*phi_poly4'*y_train4;
w_gauss4=pinv(phi_gauss4'*phi_gauss4)*phi_gauss4'*y_train4;
w_sig4=pinv(phi_sig4'*phi_sig4)*phi_sig4'*y_train4;
%error=0.5*sqrt(sum((y_train-phi*w).^2));
error_gauss4=sqrt(mean((phi_gauss4*w_gauss4-y_train4).^2));
error_sig4=sqrt(mean((phi_sig4*w_sig4-y_train4).^2));
error_poly4=sqrt(mean((phi_poly4*w_poly4-y_train4).^2));
%error_test=0.5*sqrt(sum((y_val-phi_test*w).^2));
error_test_gauss4=sqrt(mean((phi_test_gauss4*w_gauss4-y_val4).^2));
error_test_sig4=sqrt(mean((phi_test_sig4*w_sig4-y_val4).^2));
error_test_poly4=sqrt(mean((phi_test_poly4*w_poly4-y_val4).^2));

%-----------------------------total----------------------------------------
%error=0.5*sqrt(sum((y_train-phi*w).^2));
error_gauss=0.25*(error_gauss1+error_gauss2+error_gauss3+error_gauss4);
error_sig=0.25*(error_sig1+error_sig2+error_sig3+error_sig4);
error_poly=0.25*(error_poly1+error_poly2+error_poly3+error_poly4);
%error_test=0.5*sqrt(sum((y_val-phi_test*w).^2));
error_test_gauss=0.25*(error_test_gauss1+error_test_gauss2+error_test_gauss3+error_test_gauss4);
error_test_sig=0.25*(error_test_sig1+error_test_sig2+error_test_sig3+error_test_sig4);
error_test_poly=0.25*(error_test_poly1+error_test_poly2+error_test_poly3+error_test_poly4);
