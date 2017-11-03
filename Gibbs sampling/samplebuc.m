function we=samplebuc(prob,randvalue)
  count=prob(1);
  q=0;
   for i=2:size(prob,2)
    if(randvalue<=count)
       q=i-1;
       break
    endif
    count=count+prob(i);
   endfor 
   if q!=0
     we=q;
   else
   we=size(prob,2);
   endif
   endfunction