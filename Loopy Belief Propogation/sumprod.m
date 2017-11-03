function Z=sumprod(A,W,its)
      m=size(A);
      M=ones(m,m,size(W));
      BE=zeros(m,m,size(W),size(W));
      SBE=zeros(m,m);
      T=zeros(m);
      BV=zeros(m,size(W));
      SBV=zeros(m,1);
      while its>=0
        for i=1:m
          for j=1:m
            if(A(i,j)!=0)
            for c=1:size(W)
              o=0;
              for d=1:size(W)
                if(c==d)
                pot=0;
                else
                pot=1;
                endif
                NE=prne(i,M,A,d);
                h=NE*pot*exp(d);
                o=o+h;
                endfor
                M(i,j,c)=o;
                endfor
                endif
                endfor
                endfor
                T=sum(M(:,:,:),3);
                for i=1:size(W)
                  if(T !=0)
                  M(:,:,i)=M(:,:,i)./T;
                  endif
                  endfor
                  for i=1:m
                    for c=1:size(W)
                        pro=1;
                         for j=1:m
                           pro=pro*M(j,i,c);
                              endfor
                                BV(i,c)=exp(c)*pro;
                    endfor
                  endfor
                  SBV=sum(BV,2);
                  if (SBV !=0)
                  BV=BV./SBV;
                  endif
                  for i=1:m
                    for j=1:m
                       for c=1:size(W)
                           for d=1:size(W)
                             ny=edgepro(i,j,c,d,A,M);
                             BE(i,j,c,d)=ny;
                             endfor
                        endfor
                     endfor
                  endfor
                  SBE=sum(BE(:,:,:),3);
                  for i=1:size(W)
                    for j=1:size(W)
                    if (SBE !=0)
                    BE(:,:,i,j)=BE(:,:,i,j)./SBE;
                    endif
                    endfor
                    endfor
                    its--;
                   endwhile
                  
                  
                  Z=beth(BV,BE,A,size(W),m);
                  endfunction
