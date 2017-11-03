function ny=edgepro(i,j,c,d,A,M)
                      m=size(A);
                      if(c==d)
                      ny=0;
                      else
                      v=1;
                      x=1;
                      for t=1:m
                        if(t!=j)
                       v=v*M(t,i,c);
                       endif
                       endfor
                       for t=1:m
                         if(t!=i)
                         x=x*M(t,j,d);
                         endif
                         endfor
                         r=v*x*exp(c)*exp(d);
                         ny=r;
                         endif
                         endfunction
                    