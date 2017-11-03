function w = colormle(A,samples1)
  samples=samples1';
  size(samples);
  k=max(max(samples));
  gam=0.01;
  its=1000;
  temp=zeros(1,k);
  w = ones(1, k);
  for s=1:size(samples,1)
    for i=1:size(samples,2)
      j=samples(s,i);
      temp(j)=temp(j)+1;
    end
  end
      temp = temp/ size(samples, 2);
      while its > 0
      temp1=zeros(1,k);
      bn=sumprod1(A,w,10);
      for j=1:k
      temp1(j)=temp(j)-(sum(bn(:,j)));
      
      end
      
      w=w+gam*temp1;
      its = its - 1;
      end
      end
      