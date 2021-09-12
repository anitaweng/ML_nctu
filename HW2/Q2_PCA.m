%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%run Q2.m first
%PCA
N=50;
X=[input1(1:10304,:),input2(1:10304,:),input3(1:10304,:),input4(1:10304,:),input5(1:10304,:)]';
pca_mean=(1/N)*(sum(X));
pca_x=X-pca_mean;
pca_s=(1/(N-1))*(X'*X);
[V,D]=eig(pca_s);
[d,ind] = sort(diag(D),'descend');
Ds = D(ind,ind);
Vs = V(:,ind);
% pca_image_filter=(Vs(:,1)).*(X(1,:)');
% pca_image = reshape(pca_image_filter,[112,92]);

figure
pca_image = reshape(Vs(:,1),[112,92]);
pca_image =pca_image *2500;
image(pca_image);
%--------------------------------------
figure
pca_image = reshape(Vs(:,2),[112,92]);
pca_image =pca_image *2500;
image(pca_image);
%--------------------------------------
figure
pca_image = reshape(Vs(:,3),[112,92]);
pca_image =pca_image *2500;
image(pca_image);
%--------------------------------------
figure
pca_image = reshape(Vs(:,4),[112,92]);
pca_image =pca_image *2500;
image(pca_image);
%--------------------------------------
figure
pca_image = reshape(Vs(:,5),[112,92]);
pca_image =pca_image *2500;
image(pca_image);