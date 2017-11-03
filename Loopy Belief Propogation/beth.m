function Z=beth(B,BE,A,k,m)
                    sum1=0;
                    sum2=0;
                    entr=0;
                    for i=1:m
                      for j=1:k
                        sum1=sum1+(B(i,j)*log(B(i,j)));
                      endfor
                     endfor
                     for i=1:m
                       for j=1:m
                         for c=1:k
                           for d=1:k
                            if((B(i,c)!=0)&&(B(j,d)!=0))
                              y=B(i,c)*B(j,d);
                              l=BE(i,j,c,d)/y;
                              if(l>1)
                              p=BE(i,j,c,d)*log(l);
                              sum2=sum2+p;
                              endif
                              endif
                              endfor
                              endfor
                              endfor
                              endfor
                              entr=-(sum1+sum2);
                              Z=exp(entr);
                           endfunction
