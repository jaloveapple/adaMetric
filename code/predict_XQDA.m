function [ probRanks,sortedIndex] = predict_XQDA( W,M,galFea,probFea)
%PREDICT_XQDA Summary of this function goes here
%   Detailed explanation goes here
 if ~(size(galFea,1)==size(probFea,1) && size(galFea,2)==size(probFea,2))
     display('wrong data input');
 end 
 dist=MahDist(M, galFea * W,probFea * W);
 score=dist;
 [~, sortedIndex] = sort(score,'ascend');
 trueIndex=1:size(galFea,1);
 [probRanks, ~] = find( bsxfun(@eq, sortedIndex, trueIndex) );
end

