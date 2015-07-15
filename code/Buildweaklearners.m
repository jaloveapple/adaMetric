%clear;

            

featname={'feat1','feat2'};
metrics= {'XQDA','kLFDA','svmml','KISSME'};
num_wLearners=length(featname)*length(metrics);

num_itr =10; 
np_ratio =10; % The ratio of number of negative and positive pairs. Used in PCCA
% default algorithm option setting
%AlgoOption.name = algoname;
%AlgoOption.func = algoname; % note 'rPCCA' use PCCA function also.
AlgoOption.npratio = np_ratio; % negative to positive pair ratio
AlgoOption.beta =3;  % different algorithm have different meaning, refer to PCCA and LFDA paper.
%AlgoOption.d =40; % projection dimension
AlgoOption.epsilon =1e-4;
AlgoOption.lambda =0;
AlgoOption.w = [];
%AlgoOption.dataname = fname;
%AlgoOption.partitionname = partition_name;
AlgoOption.num_itr=num_itr;

%%
for i=1:length(featname)
    for j=1:length(metrics)
        index=(i-1)*length(metrics)+j;
        weakLearner{index}.featName   =featname{i}; 
       	switch weakLearner{index}.featName
            case {'feat1'}
                feat=feat1;
            case {'feat2'}
                feat=feat2;
        end
        traindata=feat(idx_train,:);
        testdata =feat(idx_test,:);
        weakLearner{index}.metricName =metrics{j};
        
        switch  weakLearner{index}.metricName
            case {'XQDA'}
                
            case {'kLFDA'}
                
                %set the kernel fof LFDA
                AlgoOption.kernel='linear';
                AlgoOption.dataname = weakLearner{index}.featName;
                AlgoOption.name=weakLearner{index}.metricName;
                AlgoOption.npratio =0; % npratio is not required.
                AlgoOption.beta =0.01; 
                AlgoOption.d =40;
                AlgoOption.LocalScalingNeighbor =6; % local scaling affinity matrix parameter.
                AlgoOption.num_itr= 10;
                
                
                
                [algo, V]= LFDA(double(traindata),gID(idx_train)' ,AlgoOption);
                [weakLearner{index}.r,weakLearner{index}.distMat]=train_result_LFDA(mat2cell(algo),mat2cell(traindata),mat2cell(traindata),idx_test_gallery,gID(idx_train));
                
            case {'svmml'}
                %%%%%%  do pca
                [COEFF,pc,latent,tsquare] = princomp(Feature,'econ');
                pcadim =  sum(cumsum(latent)/sum(latent)<0.95); %80;%
                Feature = pc(:, 1:pcadim);
                AlgoOption.doPCA = 1;
                
                AlgoOption.p = []; % learn full rank projection matrix
                AlgoOption.lambda1 = 1e-8;
                AlgoOption.lambda2 = 1e-6;
                AlgoOption.maxit = 300;
                AlgoOption.verbose = 1;
                
                AlgoOption.dataname = weakLearner{index}.featName;
                AlgoOption.name=weakLearner{index}.metricName;
                [algo] = svmml_learn_full_final(double(traindata),gID(idx_train)' ,AlgoOption);
                weakLearner{index}.distMat=train_result_svmml(algo);
            case {'KISSME'}
                [COEFF,pc,latent,tsquare] = princomp(Feature,'econ');
%         pcadim =  sum(cumsum(latent)/sum(latent)<0.95); %80;%
                pcadim=45;
                Feature = pc(:, 1:pcadim);
                AlgoOption.doPCA = 1;                
                
                
                
                AlgoOption.PCAdim = pcadim;
                AlgoOption.npratio = 10;
                AlgoOption.nFold = 20;
                AlgoOption.dataname = weakLearner{index}.featName;
                AlgoOption.name=weakLearner{index}.metricName;
                [algo] = kissme(train{c}',ix_pair,y,AlgoOption);
                weakLearner{index}.distMat=train_result_KISSME(algo,traindata,testdata,ix_partition,IDs);
        end
            
            
    end
end

%%


%%%%%%  training step


%%%%    testing step
        testdataPro=feat(idx_test,:);
        testdataGal=feat(idx_test+num_data,:);
        testdata=[testdataPro;testdataGal];