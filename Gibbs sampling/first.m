n=input('Enter the no of nodes'); 
for i=1:n
  for j=1:n
    A(i,j)=input('Enter values of Adjacency Matrix');
    endfor
    endfor
    its=input('Enter the value of iterations');
    k=input(' Enter the size  for Weights matrix');
    burnin=input('Enter the no of burn in samples');
    
    for i=1:k
      w(i)=input('Enter the weights');
      endfor
      m=gibbstest(A,w,burnin,its)
      
      
     
