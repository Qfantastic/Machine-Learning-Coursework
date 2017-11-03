function NE=prne(i,M,A,d)
                    r=1;
                    m=size(A);
                  for t=1:m
                    if(A(i,t)!=0&&t!=j)
                    r=r*M(t,i,d);
                    endif
                    endfor
                    NE=r;
                    endfunction