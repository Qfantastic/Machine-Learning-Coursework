n=input('Enter the no of nodes'); 
for i=1:n
  for j=1:n
    A(i,j)=input('Enter values of Adjacency Matrix');
    endfor
    endfor
    its=1000;
    burnin=1000;
    k=input(' Enter the size  for Weights matrix');
    for i=1:k
      we(i)=input('Enter the weights');
    endfor
    i=input('Enter the latent variable');
    samples=gibbs(A,we,burnin,its);
     w=colormle(A,samples)
     w1=colorem(A,i,samples)