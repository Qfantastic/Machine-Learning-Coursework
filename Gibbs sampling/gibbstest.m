function m=gibbs(A,w,burnin,its)
  m=zeros(size(A,1),size(w,2));
  bmatrix=zeros(1,size(A,1));
  v=1;
  k=its;
  bmatrix=randi([1,size(w,2)],1,size(A,1));
  for i=1:size(A,1)
    bmatrix(i)=w(bmatrix(i));
  endfor
  for i=1:size(A,1)
    if(v>size(w,2))
    v=1;
    endif
    bmatrix(i)=w(v);
    v++;
  endfor
  while burnin>0
        for i=1:size(A,1)
        prob=potenfn(i,w,A,bmatrix);
        randvalue=rand;
        we=samplebuc(prob,randvalue);
        bmatrix(i)=w(we);
        endfor
      burnin--;
   endwhile
   for i=1:size(bmatrix,2)
     m(i,bmatrix(i))+=1;
   endfor
   its=its-1;
   while its>0
     for i=1:size(A,1)
        prob=potenfn(i,w,A,bmatrix);
        randvalue=rand;
        we=samplebuc(prob,randvalue);
        bmatrix(i)=w(we);
     endfor
     for i=1:size(bmatrix,2)
      m(i,bmatrix(i))+=1;
     endfor
     its--;
   endwhile
   m=m./k;
   endfunction
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   