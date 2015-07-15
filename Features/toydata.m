toysize=40;
setsize=632;
% for i=1:toysize
%     gID(i)=i;
% end
% for i=1:toysize
%     gID(i+toysize)=i;
% end

parta=da(1:toysize,:);
partb=da(setsize+1:setsize+toysize,:);
feat1=[parta;partb];
save('feat1.mat','feat1');

% feat2=[parta;partb];
% save('feat2.mat','feat2');