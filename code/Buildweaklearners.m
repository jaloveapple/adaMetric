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
tic;
count=0;
%%
for i=1:length(featname)
    for j=1:length(metrics)
        index=(i-1)*length(metrics)+j;
        weakLearner{index}.featName   =featname{i}; 
       	switch weakLearner{index}.featName
            case {'feat1'}
                feat=double(feat1);
            case {'feat2'}
                feat=double(feat2);
        end
        traindata=feat(idx_train,:);
        Trnpart1=feat(idx_Trnpart1,:);
        Trnpart2=feat(idx_Trnpart2,:);
        testdata =feat(idx_test,:);
        weakLearner{index}.metricName =metrics{j};
      %%   metric choice
        switch  weakLearner{index}.metricName
            case {'XQDA'}
                AlgoOption.verbose=1;
                galFea1 = Trnpart1(1 : num_train/2, :);
                probFea1 = Trnpart1(num_train/2 + 1 : end, :);
                galFea2 = Trnpart2(1 : num_train/2, :);
                probFea2 = Trnpart2(num_train/2 + 1 : end, :);
 
                [weakLearner{index}.W1,weakLearner{index}.M1] = XQDA(galFea1, probFea1, (1:num_test/2)', (1:num_test/2)',AlgoOption);
                [ weakLearner{index}.r1,weakLearner{index}.distMat1] = predict_XQDA( weakLearner{index}.W1,weakLearner{index}.M1,galFea2,probFea2);
                [weakLearner{index}.W2,weakLearner{index}.M2] = XQDA(galFea2, probFea2, (1:num_test/2)', (1:num_test/2)',AlgoOption);
                [ weakLearner{index}.r2,weakLearner{index}.distMat2] = predict_XQDA( weakLearner{index}.W2,weakLearner{index}.M2,galFea1,probFea1);               
                
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
                
                [algo, V]= LFDA(double(Trnpart1),gID(idx_Trnpart1)' ,AlgoOption);
                [weakLearner{index}.r1,weakLearner{index}.distMat1]=train_result_LFDA(mat2cell(algo),mat2cell(Trnpart1),mat2cell(Trnpart2),idx_test_gallery,gID(idx_Trnpart2));                               
                [algo, V]= LFDA(double(Trnpart2),gID(idx_Trnpart2)' ,AlgoOption);
                [weakLearner{index}.r2,weakLearner{index}.distMat2]=train_result_LFDA(mat2cell(algo),mat2cell(Trnpart2),mat2cell(Trnpart1),idx_test_gallery,gID(idx_Trnpart1));
                
            case {'svmml'}
                %%%%%%  do pca
%                 [COEFF,pc,latent,tsquare] = princomp(Feature,'econ');
%                 pcadim =  sum(cumsum(latent)/sum(latent)<0.95); %80;%
%                 Feature = pc(:, 1:pcadim);
%                 AlgoOption.doPCA = 1;
                if(size(traindata,2)>600)
                    pcadim=600;
                    [COEFF,pc,latent,tsquare] = princomp(Trnpart1,'econ');
 %                   Trnpart1 = pc(:, 1:pcadim);
                    Trnpart1 = pc;
                    [COEFF,pc,latent,tsquare] = princomp(Trnpart2,'econ');
 %                   Trnpart2 = pc(:, 1:pcadim);                    
                    Trnpart2 = pc;
                    AlgoOption.doPCA = 1; 
                    
                    
                    
                    
                end
                
                AlgoOption.p = []; % learn full rank projection matrix
                AlgoOption.lambda1 = 1e-8;
                AlgoOption.lambda2 = 1e-6;
                AlgoOption.maxit = 300;
                AlgoOption.verbose = 0;
                
                AlgoOption.dataname = weakLearner{index}.featName;
                AlgoOption.name=weakLearner{index}.metricName;
                algo = svmml_learn_full_final(double(Trnpart1),gID(idx_Trnpart1)' ,AlgoOption);
                [weakLearner{index}.r1,weakLearner{index}.distMat1] = predict_svmml(mat2cell(algo),mat2cell(Trnpart1),mat2cell(Trnpart2),idx_test_gallery,gID(idx_Trnpart2));
                algo = svmml_learn_full_final(double(Trnpart2),gID(idx_Trnpart2)' ,AlgoOption);
                [weakLearner{index}.r2,weakLearner{index}.distMat2] = predict_svmml(mat2cell(algo),mat2cell(Trnpart2),mat2cell(Trnpart1),idx_test_gallery,gID(idx_Trnpart1));               
                
            case {'KISSME'}                
%          pcadim =  sum(cumsum(latent)/sum(latent)<0.95); %80;%
                 pcadim=45;

                [COEFF,pc,latent,tsquare] = princomp(Trnpart1,'econ');
                Trnpart1 = pc(:, 1:pcadim);
                [COEFF,pc,latent,tsquare] = princomp(Trnpart2,'econ');
                Trnpart2 = pc(:, 1:pcadim);                  
                 AlgoOption.doPCA = 1;                
                
                
                traindata=normc_safe(traindata);
                testdata =normc_safe(testdata);
                
                AlgoOption.PCAdim = pcadim;
                AlgoOption.npratio = 10;
                AlgoOption.nFold = 20;
                AlgoOption.dataname = weakLearner{index}.featName;
                AlgoOption.name=weakLearner{index}.metricName;
                %% make pairs
                
                [ix_train_pos_pair1, ix_train_neg_pair1]=GeneratePair(mpair_ID_Trnpart1);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                Nneg = min(AlgoOption.npratio* length(ix_train_pos_pair1), length(ix_train_neg_pair1));
                ix_pair1 = [ix_train_pos_pair1 ; ix_train_neg_pair1(1:Nneg,:) ]; % both positive and negative pair index
                y1 = [ones(size(ix_train_pos_pair1,1), 1); -ones(Nneg,1)]; % annotation of positive and negative pair               
 
                 [ix_train_pos_pair2, ix_train_neg_pair2]=GeneratePair(mpair_ID_Trnpart2);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                Nneg = min(AlgoOption.npratio* length(ix_train_pos_pair2), length(ix_train_neg_pair2));
                ix_pair2 = [ix_train_pos_pair2 ; ix_train_neg_pair2(1:Nneg,:) ]; % both positive and negative pair index
                y2 = [ones(size(ix_train_pos_pair2,1), 1); -ones(Nneg,1)]; % annotation of positive and negative pair                 
                %%
                algo = kissme(Trnpart1',ix_pair1,y1,AlgoOption);
                [weakLearner{index}.r1,weakLearner{index}.distMat1 ] = predict_kissme(mat2cell(algo),mat2cell(Trnpart2),idx_test_gallery,gID(idx_Trnpart2));
                algo = kissme(Trnpart2',ix_pair2,y2,AlgoOption);
                [weakLearner{index}.r2,weakLearner{index}.distMat2 ] = predict_kissme(mat2cell(algo),mat2cell(Trnpart1),idx_test_gallery,gID(idx_Trnpart1));               
               
        end
        count=count+1
        tpoint(index)=toc;
        if(index==1)
            eti=toc; 
        else
            eti=toc-tpoint(index-1);                    
        end
        display(['No.' num2str(index) 'th metric ' weakLearner{index}.metricName ' costs time ' num2str(eti) 's']);

    end
end

%%


%%%%%%  training step


