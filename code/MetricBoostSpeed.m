% Funcation Name: MetricBoostSpeed.m
% Copyright (C) 2008 - 2012.
% Created Date: Oct. 29, 2008
% Created By: Yimo Tao
% Modified By: Dijia Wu
% Modified By: Meizhu Liu
% Modified By: Tingyang Xu
% Contact: Jinbo Bi, jinbo@engr.uconn.edu
% Last Modified: Aug. 20, 2012
% Version: 0.24
% Possible modification: (1) could buffer triplet distance (xi-xj) to a
%                            2D matrix.
% Input: x              --> d*n matrix, n is number of data samples, d is
%                       the feature dimsion, i.e., each column is a feture
%                       vector
%        triplet        --> m*3 triplets of (i, j, k), representing h(xi, xk)
%                       > h(xi, xj)
%                       each row is a triplet vector, m is the number of
%                       triplets
%        param.T         --> iteration time, the iteration time also decide the
%                       number of hypothesis
%        param.htoption  --> param struct containing h_t(x,y) function option
%        param.debug     --> debug mode or not
% Output:M               --> the learned marix
%        debugInfo       --> debug information
% Usage: MetricBoost(x, triplet, T)
% ref:
% [1] A short introduction to boosting
% [2] PSDBoost: matrix generation linear programming for positive
% semidefinite matrices learnning.
% [3] Boosting of Positive Semi-defini Matrices for Metric Learning
% [4] Distance Learning for Similarity Estimation
function [M,xr,u] = MetricBoostSpeedtemp(x, triplet, param)
tic;
T = param.T;
if ~isfield(param,'belta')
    param.belta=0;
end
if ~isfield(param,'is01Norm')
    param.is01Norm=true;
end

if ~isfield(param,'speedUp')
    param.speedUp=false;
end

% output the model
if param.speedUp
    text=' with speed up';
else
    text='';
end

switch param.htoption
    case 1
        fprintf('Starting the Binary MetricBoost%s...\n',text);
    case 2
        fprintf('Starting the MetricBoost%s...\n',text);
end



% 01 normalization
if param.is01Norm
    maxX=max(x,[],2);
    minX=min(x,[],2);
    maxX=maxX-minX;
    maxX(maxX==0)=1;
    x=(x-minX*ones(1,size(x,2)))./(maxX*ones(1,size(x,2)))./2;
    
    debugInfo.maxX=maxX;
    debugInfo.minX=minX;
    clear maxX minX;
end 

% trplet errors on trainset
debugInfo.tripletErrors = zeros([1 T]);

d  =size(x,1);
m  = size(triplet,1);

xr = ComputeXrSpeed(x, triplet);
%initialize hash table
[tripletj,idj] = sortrows(triplet,[1,2]);
[tripletk,idk] = sortrows(triplet,[1,3]);
labelj = cumsum([1;any(tripletj(1:end-1,[1,2])~=tripletj(2:end,[1,2]),2)]);
labelk = cumsum([1;any(tripletk(1:end-1,[1,3])~=tripletk(2:end,[1,3]),2)]);
p = [sortrows([labelj,idj],2),sortrows([idk,labelk],1)];
p = p(:,[4,1]);
%initialize mu, D=muij*muik
mu=cell(2,1);
mu{1} = m^(-1/2)*ones(1,max(p(:,1)));
mu{2} = m^(-1/2)*ones(1,max(p(:,2)));
alpha=zeros(1,T);

for t=1:T
  
   u(t,:,:) = TrainWeakLearnerSpeed(xr,mu,p);
    %calculate error
  
   [e,r,ev,rv,rmag,h] = ComputeHyposthesisErrorSpeed(x,triplet,...
        squeeze(u(t,:,:)),mu,p);
    if (param.htoption == 2)
        u(t,:,:) = u(t,:,:)/rmag;
    end

%     alpha---speed up
    switch param.htoption
        case 1
            alpha(t) = log((1-e)/e)/2;
        case 2
            alpha(t) = log((1+r)/(1-r))/2;
    end
    % metric learning requires alpha > 0
    if (alpha < 0)
        % this means the weak learner should have less than 50% error rate
        % on the weighted training set.
        disp 'warning: alpha suppose to be larger than 0';
    end
%end alpha
    
    mu = UpdateDistributionSpeed(mu,p,h,alpha(t));
    fprintf('.');
    if ~mod(t,100), fprintf('\n');end;

end

% normalize alpha
alpha = alpha./sum(alpha(:));
% final ouput
M = MultiplyMatrix(alpha,u);
% M = M./(max(max(abs(M))));
debugInfo.M=M;

% decompose M
[L, dd] = eig( (M+M')/2 );
dd = real( diag(dd) );
% reassemble M (ignore negative eigenvalues)
dd( dd<0 ) = 0;
[ ~, IX ] = sort( dd, 'descend' );

L=L( :, IX );
dd=dd( IX );
L=( L*diag( sqrt(dd) ) )';
% M = L*L';
debugInfo.eigenvalues=dd;
debugInfo.L=L;

% intermediate output
debugInfo.alpha = alpha;
debugInfo.U = u;

toc
end

% compute the column vector of rank one matrix
% Input:  ev         --> error indicator vector of the triplet
%         D          --> distribution
%         alpha      --> parameter
% Output: newD       --> newD
function mu = UpdateDistributionSpeed(mu,p,h,alpha)
        mu{1} = mu{1}.*exp(-alpha*h{1});
        mu{2} = mu{2}.*exp(alpha*h{2});
        Z=sum(accumarray(p(:,1),mu{2}(p(:,2))).*mu{1}(:));
        mu{1}=mu{1}*Z^(-0.5);
        mu{2}=mu{2}*Z^(-0.5);
end


% compute hypothesis error
% Input:  x          --> feature vectors
%         triplet    --> triplets
%         U          --> rank one marix
%         D          --> distribution
% Output: e          --> error
%         ev         --> error indicator vector
function [e,r,ev,rv,rmag,h] = ComputeHyposthesisErrorSpeed(x, triplet, U, mu,p)

pairs{1} = unique(triplet(:,[1,3]),'rows');
pairs{2} = unique(triplet(:,[1,2]),'rows');

hij=INNERPM(x,pairs{2}(:,1),pairs{2}(:,2),U)';
hik=INNERPM(x,pairs{1}(:,1),pairs{1}(:,2),U)';
h=cell(2,1);
h{1}=hik;
h{2}=hij;
rv=hij(p(:,2))-hik(p(:,1));
ev=rv>=0;%param.belta;
e=sum(accumarray(p(ev,1),mu{2}(p(ev,2)),[max(p(:,1)) 1]).*mu{1}(:));

rmag = max(abs(rv));
h{1}=h{1}/rmag;
h{2}=h{2}/rmag;
rv = rv/rmag;

r=-sum(accumarray(p(:,1),rv.*mu{2}(p(:,2))).*mu{1}(:));

end


% train weak learner
% Input:     xr   --> xr marix
%            D    --> distribution
% Output:    U    --> weak distance learner
% Maximize u'*(D*Xr)*u
function [U] = TrainWeakLearnerSpeed(xr,mu,p)
[a] = MultiplyMatrixSpeed(mu,xr,p);

a=reshape(squeeze(a),size(xr{1},2),size(xr{1},2));
opts.disp   = 0;
opts.issym  = 1;
opts.isreal = 1;

% note that eig output the unit vector
[eigVect,~] = eigs((a+a')/2,1,'la',opts);
% use the largest eig
u = eigVect;
u = NormalizeColumnToUnitVect(u);
U = u*u';

end



function xr = ComputeXrSpeed(x, triplet)
xr=cell(2,1);
p=unique(triplet(:,[1,3]),'rows');

xik=(x(:,p(:,1))-x(:,p(:,2)))';
xr{1}=xik;

p=unique(triplet(:,[1,2]),'rows');
xij=(x(:,p(:,1))-x(:,p(:,2)))';
xr{2}=xij;

end


% multiply matrix
% Input: a  --> 1*n matrix
%        b  --> n*m*m matrix
%        rs --> m*m matrix
function rs = MultiplyMatrix(a,b)
rs = zeros([size(b,2) size(b,3)]);
for i=1:length(a)
    rs = rs + a(i).*squeeze(b(i,:,:));
end
end

function [rs] = MultiplyMatrixSpeed(a,b,p)
sumij = accumarray(p(:,2),a{1}(p(:,1))).*a{2}(:);
sumik = accumarray(p(:,1),a{2}(p(:,2))).*a{1}(:);
xij = mulh(b{2}',sumij);
xik = mulh(b{1}',sumik);
rs = xik*b{1} - xij*b{2};
end

% normalize column vect to univect
function vect = NormalizeColumnToUnitVect(srcVect)
%normalize by the first strenght

for i=1:size(srcVect,2)
    a = squeeze(srcVect(:,i));
    pos = find(abs(a) > eps, 1, 'first');
    
    if isempty(pos)
        disp( 'warning: possible error in normalization, a zero vector');
    end
    
    a = a./a(pos);%normalize the sign
    a = a./sqrt(sum(a.^2));%normalize the scale
    vect(:,i) = a;
end

end