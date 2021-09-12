%readfile
filename = 'Dataset/dataset_X.csv';
x = xlsread(filename);
filename = 'Dataset/dataset_T.csv';
y = xlsread(filename);
x_train=x(1:768,:);
x_val=x(769:1096,:);
y_train=y(1:768,:);
y_val=y(769:1096,:);
%[m,n]=size(x_train);

%normalize data
% x_train=(x_train-mean(x_train)).*(1./std(x_train));
% y_train=(y_train-mean(y_train)).*(1./std(y_train));
% x_val=(x_val-mean(x_val)).*(1./std(x_val));
% y_val=(y_val-mean(y_val)).*(1./std(y_val));

%produce phi1
phi1=[ones(size(x_train,1),1),x_train];
%calculate w
w=pinv(phi1'*phi1)*phi1'*y_train;
error1=0.5*(1/size(x_train,1))*sum((phi1*w-y_train).^2);
error1_rms=sqrt(mean((phi1*w-y_train).^2));
%test set
phi1_test=[ones(size(x_val,1),1),x_val];
error1_test=0.5*(1/size(x_val,1))*sum((phi1_test*w-y_val).^2);
error1_test_rms=sqrt(mean((phi1_test*w-y_val).^2));
%produce phi2
% phi2=zeros(size(x_train,1),1+size(x_train,2)+size(x_train,2)^2);
% for i=1:size(x_train,1)
%    order2=reshape(x_train(i,:)'*x_train(i,:),1,[]);
%    phi2(i,:)=[1,x_train(i,:),order2];
% end
phi2=zeros(768,171);
order2=zeros(1,153);
for i=1:size(x_train,1)
    n=0;
    for j = 1:size(x_train,2)
        for k = j:size(x_train,2)
            n=n+1;
            order2(n)=x_train(i,j)*x_train(i,k);
        end
    end
    phi2(i,:)=[1,x_train(i,:),order2];
end
%calculate w
w=pinv(phi2'*phi2)*phi2'*y_train;
error2=0.5*(1/size(x_train,1))*sum((phi2*w-y_train).^2);
error2_rms=sqrt(mean((phi2*w-y_train).^2));
%test set
% phi2_test=zeros(size(x_val,1),1+size(x_train,2)+size(x_train,2)^2);
% for i=1:size(x_val,1)
%    order2=reshape(x_val(i,:)'*x_val(i,:),1,[]);
%    phi2_test(i,:)=[1,x_val(i,:),order2];
% end 
phi2_test=zeros(328,171);
order2=zeros(1,153);

for i=1:size(x_val,1)
    n=0;
    for j = 1:size(x_val,2)
        for k = j:size(x_val,2)
            n=n+1;
            order2(n)=x_val(i,j)*x_val(i,k);
        end
    end
    phi2_test(i,:)=[1,x_val(i,:),order2];
end
error2_test=0.5*(1/size(x_val,1))*sum((phi2_test*w-y_val).^2);
error2_test_rms=sqrt(mean((phi2_test*w-y_val).^2));

%measure the most contributed attribute
%remove 1
phi1_1=[ones(size(x_train,1),1),x_train(:,2:17)];
w=pinv(phi1_1'*phi1_1)*phi1_1'*y_train;
error1_1=0.5*(1/size(x_train,1))*sum((phi1_1*w-y_train).^2);
error1_1_rms=sqrt(mean((phi1_1*w-y_train).^2));
%remove 2
phi1_2=[ones(size(x_train,1),1),[x_train(:,1),x_train(:,3:17)]];
w=pinv(phi1_2'*phi1_2)*phi1_2'*y_train;
error1_2=0.5*(1/size(x_train,1))*sum((phi1_2*w-y_train).^2);
error1_2_rms=sqrt(mean((phi1_2*w-y_train).^2));
%remove 3
phi1_3=[ones(size(x_train,1),1),[x_train(:,1:2),x_train(:,4:17)]];
w=pinv(phi1_3'*phi1_3)*phi1_3'*y_train;
error1_3=0.5*(1/size(x_train,1))*sum((phi1_3*w-y_train).^2);
error1_3_rms=sqrt(mean((phi1_3*w-y_train).^2));
%remove 4
phi1_4=[ones(size(x_train,1),1),[x_train(:,1:3),x_train(:,5:17)]];
w=pinv(phi1_4'*phi1_4)*phi1_4'*y_train;
error1_4=0.5*(1/size(x_train,1))*sum((phi1_4*w-y_train).^2);
error1_4_rms=sqrt(mean((phi1_4*w-y_train).^2));
%remove 5
phi1_5=[ones(size(x_train,1),1),[x_train(:,1:4),x_train(:,6:17)]];
w=pinv(phi1_5'*phi1_5)*phi1_5'*y_train;
error1_5=0.5*(1/size(x_train,1))*sum((phi1_5*w-y_train).^2);
error1_5_rms=sqrt(mean((phi1_5*w-y_train).^2));
%remove 6
phi1_6=[ones(size(x_train,1),1),[x_train(:,1:5),x_train(:,7:17)]];
w=pinv(phi1_6'*phi1_6)*phi1_6'*y_train;
error1_6=0.5*(1/size(x_train,1))*sum((phi1_6*w-y_train).^2);
error1_6_rms=sqrt(mean((phi1_6*w-y_train).^2));
%remove 7
phi1_7=[ones(size(x_train,1),1),[x_train(:,1:6),x_train(:,8:17)]];
w=pinv(phi1_7'*phi1_7)*phi1_7'*y_train;
error1_7=0.5*(1/size(x_train,1))*sum((phi1_7*w-y_train).^2);
error1_7_rms=sqrt(mean((phi1_7*w-y_train).^2));
%remove 8
phi1_8=[ones(size(x_train,1),1),[x_train(:,1:7),x_train(:,9:17)]];
w=pinv(phi1_8'*phi1_8)*phi1_8'*y_train;
error1_8=0.5*(1/size(x_train,1))*sum((phi1_8*w-y_train).^2);
error1_8_rms=sqrt(mean((phi1_8*w-y_train).^2));
%remove 9
phi1_9=[ones(size(x_train,1),1),[x_train(:,1:8),x_train(:,10:17)]];
w=pinv(phi1_9'*phi1_9)*phi1_9'*y_train;
error1_9=0.5*(1/size(x_train,1))*sum((phi1_9*w-y_train).^2);
error1_9_rms=sqrt(mean((phi1_9*w-y_train).^2));
%remove 10
phi1_10=[ones(size(x_train,1),1),[x_train(:,1:9),x_train(:,11:17)]];
w=pinv(phi1_10'*phi1_10)*phi1_10'*y_train;
error1_10=0.5*(1/size(x_train,1))*sum((phi1_10*w-y_train).^2);
error1_10_rms=sqrt(mean((phi1_10*w-y_train).^2));
%remove 11
phi1_11=[ones(size(x_train,1),1),[x_train(:,1:10),x_train(:,12:17)]];
w=pinv(phi1_11'*phi1_11)*phi1_11'*y_train;
error1_11=0.5*(1/size(x_train,1))*sum((phi1_11*w-y_train).^2);
error1_11_rms=sqrt(mean((phi1_11*w-y_train).^2));
%remove 12
phi1_12=[ones(size(x_train,1),1),[x_train(:,1:11),x_train(:,13:17)]];
w=pinv(phi1_12'*phi1_12)*phi1_12'*y_train;
error1_12=0.5*(1/size(x_train,1))*sum((phi1_12*w-y_train).^2);
error1_12_rms=sqrt(mean((phi1_12*w-y_train).^2));
%remove 13
phi1_13=[ones(size(x_train,1),1),[x_train(:,1:12),x_train(:,14:17)]];
w=pinv(phi1_13'*phi1_13)*phi1_13'*y_train;
error1_13=0.5*(1/size(x_train,1))*sum((phi1_13*w-y_train).^2);
error1_13_rms=sqrt(mean((phi1_13*w-y_train).^2));
%remove 14
phi1_14=[ones(size(x_train,1),1),[x_train(:,1:13),x_train(:,15:17)]];
w=pinv(phi1_14'*phi1_14)*phi1_14'*y_train;
error1_14=0.5*(1/size(x_train,1))*sum((phi1_14*w-y_train).^2);
error1_14_rms=sqrt(mean((phi1_14*w-y_train).^2));
%remove 15
phi1_15=[ones(size(x_train,1),1),[x_train(:,1:14),x_train(:,16:17)]];
w=pinv(phi1_15'*phi1_15)*phi1_15'*y_train;
error1_15=0.5*(1/size(x_train,1))*sum((phi1_15*w-y_train).^2);
error1_15_rms=sqrt(mean((phi1_15*w-y_train).^2));
%remove 16
phi1_16=[ones(size(x_train,1),1),[x_train(:,1:15),x_train(:,17)]];
w=pinv(phi1_16'*phi1_16)*phi1_16'*y_train;
error1_16=0.5*(1/size(x_train,1))*sum((phi1_16*w-y_train).^2);
error1_16_rms=sqrt(mean((phi1_16*w-y_train).^2));
%remove 17
phi1_17=[ones(size(x_train,1),1),[x_train(:,1:16)]];
w=pinv(phi1_17'*phi1_17)*phi1_17'*y_train;
error1_17=0.5*(1/size(x_train,1))*sum((phi1_17*w-y_train).^2);
error1_17_rms=sqrt(mean((phi1_17*w-y_train).^2));
