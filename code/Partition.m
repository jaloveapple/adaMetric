%%%%%%%%%%%%%partitioning data into training set and test set
num_train=num_person/2;
num_test=num_person/2;
idxtemp=randperm(num_person);
idx_test=idxtemp(1:num_test);
idx_train=idxtemp(num_test+1:end);
mpair_ID_train=[unique(idx_train),unique(idx_train)];


idx_test=[idx_test,idx_test+num_person];
idx_train=[idx_train,idx_train+num_person];



progalrand=logical(ceil(2*rand(1,num_test))-1);
idx_test_gallery=[progalrand,~progalrand];
