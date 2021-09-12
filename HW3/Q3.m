img=imread('hw3_3.jpeg');
imaged=im2double(img);%normalize image

k=10; 
mu=rand(k,3);
% mu_table(1,:)=mu;
x=reshape(imaged,[],3);
r=zeros(size(x,1),k);
dist=zeros(size(x,1),k);
iter=2;
while 1
    for i = 1:size(x,1)
        for j = 1:k   
             dist(i,j)=sum((x(i,:)-mu(j,:)).^2);   
        end   
        [m,index]=min(dist(i,:));   
        r(i,index)=1;   
    end        
    
    loss=sum(sum(r.*dist));
    for i=1:k
       mu(i,:)=sum(r(:,i).*x)/sum(r(:,i));
    end
    mu_table(:,:,iter)=mu;
    iter=iter+1;
    if iter>100
        break;
    end 
end
%initial
co_var=zeros(k,1);
for i=1:k
    summ=0;
    for j=1:size(x,1)
        summ=summ+r(j,i)*(x(j,:)-mu(i,:))*(x(j,:)-mu(i,:))';
    end
    summ=summ/sum(r(:,i));
    co_var(i,1)=summ;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%em
% ML=zeros(100,3);
% 

pi=sum(r)/size(x,1);
% for i = 1:100
%     for j=1:k
%         %e step
% %         temppp=zeros(size(x,1),1);
%         for n=1:size(x,1)
%             temppp=0;
%             temp(n)=sum(normrnd(mu(j,:),sqrt(co_var(j,:))),2);
%             for m=1:k
%                 temppp=temppp+pi(m)*normrnd(mu(m,:),sqrt(co_var(m,:)));
%             end
%             tempp(n)=sum(temppp,2);
%         end
%         r(:,j)=temp./tempp;
%         %m step
%         for m=1:k
%             mu(m,:)=sum(r(:,m).*x)/sum(r(:,m));
%         end
%         for m=1:k
%             summ=0;
%             for n=1:size(x,1)
%                 summ=summ+r(n,m)*(x(n,:)-mu(m,:))*(x(n,:)-mu(m,:))';
%             end
%             summ=summ/sum(r(:,m));
%             co_var(m,1)=summ;
%         end
%         pi=sum(r)/size(x,1);
%         %ML(i,1)=sum(log(sum(pi.*temp,2)));
%     end
% end

%draw
% figure
% hold on
% plot(1:100,ML) 
% title('The log likelihood curve of GMM')
% hold off

% [drawm1,draw_index1]=min(abs((x(:,1)-mu(:,1)')),[],2);
% [drawm2,draw_index2]=min(abs((x(:,2)-mu(:,2)')),[],2);
% [drawm3,draw_index3]=min(abs((x(:,3)-mu(:,3)')),[],2);
pic=zeros(size(x,1),3);
for i = 1:size(x,1)
        for j = 1:k   
             dist(i,j)=sum((x(i,:)-mu(j,:)).^2);   
        end   
        [m,index]=min(dist(i,:));   
        pic(i,:)=mu(index,:);   
end   

% temp_draww1=zeros(1,size(x,1));
% temp_draww2=zeros(1,size(x,1));
% temp_draww3=zeros(1,size(x,1));
% for i=1:size(x,1)
%     t1=mu(draw_index1(i),1);
%     t2=mu(draw_index2(i),2);
%     t3=mu(draw_index3(i),3);
%     temp_draww1(i)=real(t1);
%     temp_draww2(i)=real(t2);
%     temp_draww3(i)=real(t3);
% end
figure
% pic=reshape([temp_draww1';temp_draww2';temp_draww3'],[344,500,3]);
picc=reshape(pic,[344,500,3]);
image(picc);
