featureDir = '../Features/';
addpath(featureDir);
load('feat1.mat');
load('feat2.mat');

%toysize=40;
for i=1:num_person
    gID(i)=i;
end
for i=1:num_person
    gID(i+num_person)=i;
end
assert(2*num_person==size(feat1,1));
assert(2*num_person==size(feat2,1));