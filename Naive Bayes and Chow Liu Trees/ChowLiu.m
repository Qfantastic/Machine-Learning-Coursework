function[tree] = ChowLiu
   dataval = textread('student_train.data', "%s", "delimiter", "\n");
   rows = size(dataval,1);
  for i= 1:rows
    dataval{i,1} = strsplit(dataval{i,1}, ',');
  end
  
   [model,prior] = NBayes();
   sze = size(model, 2);
   mis = zeros(sze, sze);
   disp(sze);
   for i = 1:sze
     for j = i+1:sze
       xi = model{1,i};
       xj = model{1,j};
       mis(i,j) = Mutual(xi, i, xj, j, dataval);
       mis(j,i) = mis(i,j);
     end
   end
   

   tree = minspantree(mis);
       disp(size(tree));

end
function[k] = Mutual(xi,u, xj, v, dataval)
   
   f1 = fieldnames(xi);
   f2 = fieldnames(xj);
  
   m = size(dataval,1)-1;       
   k = 0;
   for i=1:size(f1,1)
     for j=1:size(f2,1)
       c=  countval(f1{i}, u, f2{j}, v, dataval); 
       c1 = countval1(f1{i}, u, dataval);
       c2 = countval1(f2{j}, v, dataval);
      
       if(c!=0)
         k += c/m * log(c*m/(c1*c2));
        end
     end
   end
end
function [c] = countval(val1, u, val2, v, dataval)
  
   rows = size(dataval,1);
   c = 0;
   for i = 2:rows-1;
     o1 = dataval{i,1}{1,u};
     o2 = dataval{i,1}{1,v};
     if (strcmp(o1,val1) && strcmp(o2,val2) )
       c += 1;
     end
   end
end
function [c] = countval1(val1, u,dataval)
   rows = size(dataval,1);
   c = 0;
   for i = 1:rows-1
     o1 = dataval{i,1}{1,u};
     if( strcmp(o1, val1))
       c += 1;
     end
   end
end
