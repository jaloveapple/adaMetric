function [ R,Alldist,ixx] = predict_svmml( Method,train,test,ix_partition, IDs)
%PREDICT_SVMML Summary of this function goes here
%   Detailed explanation goes here
%        [r, dis] = compute_rank_svmml(algo,train,test,ix_test_gallery, gID(idx_test))

    for k = 1:size(ix_partition,1)
        ix_ref = ix_partition(k,:) == 1;
        if min(min(double(ix_partition))) < 0
            ix_prob = ix_partition(k,:) ==-1; 
        else
            ix_prob = ix_partition(k,:) ==0;
        end
        ref_ID = IDs(ix_ref);
        prob_ID = IDs(ix_prob);

        dis = 0;
        for c = 1:numel(test)
            A = Method{c}.A;
            B = Method{c}.B;
            b = Method{c}.b;
            K_test = test{c}';
            K_ref = K_test(:, ix_ref);
            K_prob = K_test(:, ix_prob);


            max_dim = max(sum(ix_prob),sum(ix_ref));
            f1 = zeros(max_dim);
            f2 = zeros(max_dim);
            f3 = zeros(max_dim);
            f1 = 0.5*repmat(diag(K_prob'*A*K_prob),[1,sum(ix_ref)]);
            f2 = 0.5*repmat(diag(K_ref'*A*K_ref)',[sum(ix_prob),1]);       
            f3 = K_prob'*B*K_ref;
            dis = dis+f1+f2-f3+b;
        end

        for i = 1:size(K_prob,2)
            [tmp, ix] = sort(dis(i, :));
            r(i) =  find(ref_ID(ix) == prob_ID(i));
            ixx(i,:)=ix;
        end
        R(k, :) = r; 
        Alldist{k} = dis; % distance matrix
    end

end

