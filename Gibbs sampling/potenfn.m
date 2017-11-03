function prob=potenfn(i,w,A,bmatrix)
  prob=zeros(1,size(w,2));
  r=bmatrix(i);
  for j=1:size(w,2)
    pot=1;
    edgepot=1;
    bmatrix(i)=w(j);
    for k=1:size(bmatrix,2)
      pot=pot*exp(bmatrix(k));
    endfor
    for c=1:size(A,1)
      for d=(c+1):size(A,1)
        if(A(c,d)==1)
          if(bmatrix(c)==bmatrix(d))
            edgepot=0;
          endif
        endif
       endfor
     endfor
     prob(j)=edgepot*pot;
    endfor
    bmatrix(i)=r;
    s=sum(prob,2);
    if(s!=0)
    prob=prob/s;
    else
    prob=ones(1,size(w,2));
    s=sum(prob,2);
    prob=prob/s;
    endif
    endfunction
    
            
    
    
    