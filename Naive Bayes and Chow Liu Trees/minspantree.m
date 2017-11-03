

function tr = minspantree(adj)

adj = -1*adj;
n = length(adj); 
tr = zeros(n);   

adj(find(adj==0))=inf; 

cnodes = 1;        
rem_nodes = [2:n];     

while length(rem_nodes)>0
  
  [minlink]=min(min(adj(cnodes,rem_nodes)));
  ind=find(adj(cnodes,rem_nodes)==minlink);

  [ind_i,ind_j] = ind2sub([length(cnodes),length(rem_nodes)],ind(1));

  i=cnodes(ind_i); j=rem_nodes(ind_j); 
  tr(i,j)=1; tr(j,i)=1;
  cnodes = [cnodes j];
  rem_nodes = setdiff(rem_nodes,j);
  
end