featureDir = '../Features/';
addpath(featureDir);
load('feat1.mat');
load('feat2.mat');

%toysize=40;
for i=1:toysize
    gID(i)=i;
end
for i=1:toysize
    gID(i+toysize)=i;
end
assert(2*toysize==size(feat1,1));
assert(2*toysize==size(feat2,1));