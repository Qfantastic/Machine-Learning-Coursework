function q=findq(i,sample,wp,k)
  q=zeros(1,k);
  wq=zeros(1,size(sample,2));
  for p=1:size(wq,2)
    if(p!=i)
    wq(p)=wp(sample(p));
    elseif(p==i)
    wq(p)=0;
    end
  endfor
  for e=1:k
    q(e)=exp(sum(wq)+wp(e));
  endfor
  sqe=sum(q);
  q=q/sqe;
endfunction
    