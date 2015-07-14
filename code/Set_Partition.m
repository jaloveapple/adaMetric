%%%%%%%%%%%%%partitioning data into training set and test set
num_test=10;
num_data=20;

idxtemp=randperm(num_data);
idx_test=idxtemp(1:num_test);
idx_train=idxtemp(num_test+1:end);


