%%%%%%%%%%%%%partitioning data into training set and test set
num_train=num_person/2;
num_test=num_person/2;
idxtemp=randperm(num_person);
idx_test=idxtemp(1:num_test);
idx_train=idxtemp(num_test+1:end);
%mpair_ID_train=[unique(idx_train),unique(idx_train)];
idx_Trnpart1=idx_train(1:num_test/2);
idx_Trnpart2=idx_train(num_test/2+1:end);

mpair_ID_Trnpart1=[unique(idx_Trnpart1),unique(idx_Trnpart1)];
mpair_ID_Trnpart2=[unique(idx_Trnpart2),unique(idx_Trnpart1)];

idx_test=[idx_test,idx_test+num_person];
idx_train=[idx_train,idx_train+num_person];
idx_Trnpart1=[idx_Trnpart1,idx_Trnpart1+num_person];
idx_Trnpart2=[idx_Trnpart2,idx_Trnpart2+num_person];


% progalrand=logical(ceil(2*rand(1,num_test))-1);
% idx_test_gallery=[progalrand,~progalrand];
progalrand=logical(ceil(2*rand(1,num_test/2))-1);
idx_test_gallery=[progalrand,~progalrand];
