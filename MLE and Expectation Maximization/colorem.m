function w=colorem(A,i,samples1)
  samples=samples1';
  size(samples);
  k=max(max(samples));
  gam=0.01;
  its=1000;
  temp=zeros(1,k);
  timesteps=10;
  
  for s=1:size(samples,1)
    for i=1:size(samples,2)
      j=samples(s,i);
      temp(j)=temp(j)+1;
    endfor
  endfor
  temp = temp/ size(samples, 2);
      wp=randi(k,1,k);
      while timesteps>0
        w = randi(k,1, k);
        while its > 0
          temp1=zeros(1,k);
          bn=sumprod1(A,w,10);
          for p=1:k
            for t=1:size(samples,1)
              q=findq(i,samples(t,:),wp,k);
              for d=1:k
                temp1(p)=q(k)*(temp(p)-sum(bn(:,p)));
              endfor
            endfor
          endfor
          w=w+gam*temp1;
          its--;
          endwhile
        wp=w;
        timesteps = timesteps-1;
        endwhile
        endfunction
        
        
       
      