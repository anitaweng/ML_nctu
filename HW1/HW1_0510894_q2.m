%readfile
filename = 'Dataset/dataset_X.csv';
x = xlsread(filename);
filename = 'Dataset/dataset_T.csv';
y = xlsread(filename);
x_train=x(1:768,:);
x_val=x(769:1096,:);
y_train=y(1:768,:);
y_val=y(769:1096,:);
x_train_trim=x(1:768,[3,8,9,11,14]);
x_val_trim=x(769:1096,[3,8,9,11,14]);
y_train_trim=y(1:768,:);
y_val_trim=y(769:1096,:);

%normalize data
% x_train=(x_train-mean(x_train)).*(1./std(x_train));
% y_train=(y_train-mean(y_train)).*(1./std(y_train));
% x_val=(x_val-mean(x_val)).*(1./std(x_val));
% y_val=(y_val-mean(y_val)).*(1./std(y_val));
%%%%%%%%%%%%%%%%%%%%%%%%% trim data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phi_temp=zeros(768,32);
order2=zeros(1,15);

for i=1:size(x_train_trim,1)
    n=0;
    for j = 1:size(x_train_trim,2)
        for k = j:size(x_train_trim,2)
            n=n+1;
            order2(n)=x_train_trim(i,j)*x_train_trim(i,k);
        end
    end
    phi_temp(i,:)=[x_train(i,:),order2];
end

phi2_test_temp=zeros(328,32);
order2=zeros(1,15);

for i=1:size(x_val_trim,1)
    n=0;
    for j = 1:size(x_val_trim,2)
        for k = j:size(x_val_trim,2)
            n=n+1;
            order2(n)=x_val_trim(i,j)*x_val_trim(i,k);
        end
    end
    phi2_test_temp(i,:)=[x_val(i,:),order2];
end
%%%%%%%%%%%%%%%%%%%%%%%% whole data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% phi_temp=zeros(768,170);
% order2=zeros(1,152);
% 
% for i=1:size(x_train,1)
%     n=0;
%     for j = 1:size(x_train,2)
%         for k = j:size(x_train,2)
%             n=n+1;
%             order2(n)=x_train(i,j)*x_train(i,k);
%         end
%     end
%     phi_temp(i,:)=[x_train(i,:),order2];
% end
% 
% phi2_test_temp=zeros(328,170);
% order2=zeros(1,152);
% 
% for i=1:size(x_val,1)
%     n=0;
%     for j = 1:size(x_val,2)
%         for k = j:size(x_val,2)
%             n=n+1;
%             order2(n)=x_val(i,j)*x_val(i,k);
%         end
%     end
%     phi2_test_temp(i,:)=[x_val(i,:),order2];
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mu=mean(x_train);
% mu_test=mean(x_val);
% sigma=std(x_train);
% sigma_test=std(x_val);
mu=mean(phi_temp);
mu_test=mean(phi2_test_temp);
sigma=std(phi_temp);
sigma_test=std(phi2_test_temp);

phi_poly=[ones(size(phi_temp,1),1),phi_temp];
phi_gauss=exp(-(phi_temp-mu).^2./(2*sigma.^2));
phi_gauss=[ones(size(phi_temp,1),1),phi_gauss];
phi_sig=1./(1.+exp(-(phi_temp-mu)./sigma));
phi_sig=[ones(size(phi_temp,1),1),phi_sig];
phi_test_gauss=exp(-(phi2_test_temp-mu_test).^2./(2*sigma_test.^2));
phi_test_gauss=[ones(size(phi2_test_temp,1),1),phi_test_gauss];
phi_test_sig=1./(1.+exp(-(phi2_test_temp-mu_test)./sigma_test));
phi_test_sig=[ones(size(phi2_test_temp,1),1),phi_test_sig];
phi_test_poly=[ones(size(phi2_test_temp,1),1),phi2_test_temp];
w_poly=pinv(phi_poly'*phi_poly)*phi_poly'*y_train;
w_gauss=pinv(phi_gauss'*phi_gauss)*phi_gauss'*y_train;
w_sig=pinv(phi_sig'*phi_sig)*phi_sig'*y_train;
%error=0.5*sqrt(sum((y_train-phi*w).^2));
error_gauss=sqrt(mean((phi_gauss*w_gauss-y_train).^2));
error_sig=sqrt(mean((phi_sig*w_sig-y_train).^2));
error_poly=sqrt(mean((phi_poly*w_poly-y_train).^2));
%error_test=0.5*sqrt(sum((y_val-phi_test*w).^2));
error_test_gauss=sqrt(mean((phi_test_gauss*w_gauss-y_val).^2));
error_test_sig=sqrt(mean((phi_test_sig*w_sig-y_val).^2));
error_test_poly=sqrt(mean((phi_test_poly*w_poly-y_val).^2));


