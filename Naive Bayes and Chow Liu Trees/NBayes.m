function [model, accuracy] = NBayes ()
  dataval = textread('student_train.data', "%s", "delimiter", "\n");
  rows = size(dataval,1);
  for i= 1:rows
    dataval{i,1} = strsplit(dataval{i,1}, ',');
  end
  rows = size(dataval,1);
  params = size(dataval{1,1},2)-1;
  model = cell(1,params);
  ngrades = 21;
  gradecolumns = 19;
  
  initializer = ones(1,ngrades);
  priorvalue = zeros(1,ngrades);
  for i= 2:rows-1
    cls = dataval{i,1}{1,gradecolumns};
    clsv = str2double(cls) + 1;  
    priorvalue(1,clsv) += 1;
    for j = 1:params
       val = dataval{i,1}{1,j};
       
       if ~isfield(model{j}, val)
          model{j}.(val) = initializer;
       endif
        model{j}.(val)(clsv) += 1;
    end
  end
  
  priorvalue = priorvalue / sum(priorvalue(:));
 
  
  
  
  test_data = textread('student_test.data', "%s", "delimiter", "\n");
  
  rows = size(test_data,1);
  for i= 1:rows
    test_data{i,1} = strsplit(test_data{i,1}, ',');
  end

  
   rows = size(test_data,1);
  params = size(test_data{1,1},2) -1;
  gradecolumns = 19;
  ngrades = 21;
  
  output = zeros(1,rows);
  cnt = 0;
  
  for i = 2:rows
    dist = ones(1,ngrades);
    for j = 1:params
      val = test_data{i,1}{1,j};
     
      for k = 1:ngrades
          
          try
            dist(1,k) = dist(1,k)*condprod(model{j}, val, k);
          catch
          
          end
        
      end
    end

    prd = dist.*priorvalue;
    prd = prd / sum(prd(:));

    [M, I] = max(prd);
    output(1,i) = I -1;
  
    org = str2double(test_data{i,1}{1,gradecolumns});
    if ( org == output(1,i))
      cnt += 1;
    end
 
   end
   
   
  
     rows = size(test_data,1);
 
  gradecolumns = 19;
  pscore = 0;
  nscore = 0;
  
  for i =2:rows
    org = str2double(test_data{i,1}{1,gradecolumns});
    out = output(i);
    if(org == out)
      pscore += 1;
    else
      nscore += 1;
    end
  end
   disp('Accuracy:');
   accuracy = (pscore/(pscore+nscore)); 
   

   
end

function[P] = condprod(param, val, cls)
  names = fieldnames(param);
  n = size(names,1);
  count = 0;
  initcount = param.(val)(cls);
  for i = 1:n
    count += param.(names{i})(cls);
  end

  if(initcount == 0)
    P = 1/n;
    return;
  end
  P= initcount / count;
  
endfunction
