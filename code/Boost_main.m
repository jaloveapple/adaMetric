preset;
Buildweaklearners;

iterations=16;
% wm means sample weights
wm=2/num_test*ones(1,num_test/2);
for i=1:iterations
 % calculate the error rates
   [ Em,id_weak(i),class_result ]=calcu_errors(weakLearner,wm,mod(i,2));
  %find the best one
  if min(Em)>0.5
      error('weak classifier wrong');
      break;
  end
   %[minerr,id_weak(i)]=min(Em); 
   % updata  alpha the weight of weakclassifier
   alpha(i)=0.5*log2((1-Em)/Em);%%%%    testing step
  %  updata w  the weight of samples 
   Zm=wm*exp(-alpha(i)*class_result);
   wm=wm/Zm.*exp(-alpha(i)*class_result)';
    
    
end
  
    
%%%%    testing step
