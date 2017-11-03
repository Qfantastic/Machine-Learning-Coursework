function bn=sumprod1(A,w,its)
[~,colors]=size(w);
[n,~]=size(A);

%messages will be a 3-d matrix of the 
%2 nodes and the color of the destination
message=ones(n,n,colors)*rand(); %stores the messages at time t-1
future_message=zeros(n,n,colors); %stores the message at time t    
    %calculating the messages
    for t=1:its
        for i=1:n
            for j=1:n
                %stores the normalizing constant from i to j at time t
                normalize=0;             
                for k=1:colors
                    for l=1:colors%the psi function correspods to whether the color of the two edges  are same or not
                        future_message(i,j,k)= future_message(i,j,k)+exp(w(l))*sign(abs(l-k))*getProduct1(A,message,i,j,l);
                    end
                    normalize=normalize+future_message(i,j,k);
                end          
                %normalizing the message from i to j at time t
                for k=1:colors
                    future_message(i,j,k)= future_message(i,j,k)/normalize;
                end 
            end
        end
        %setting the previous matrix as the current matrix
        %and making the current matrix as 0
        message=future_message;
        future_message=zeros(n,n,colors);
    end
    
    %calculating the marginal distribution of the vertices
    bn=zeros(n,colors);
    for i=1:n
        normal=0; %normalizing constant
        for k=1:colors
            bn(i,k)=exp(w(k))*getProduct2(A,message,i,k);
            normal=normal+bn(i,k);
        end
        %normalizing the marginal distributions
        for k=1:colors
            bn(i,k)=bn(i,k)/normal;
        end     
    end
    %disp(bn);
end

%A-adjacency matrix
%message-message matrix
%source - the vertex which is not to be included in the multiplication
%dest - the vertex to which the messages are to be multiplied
%color - the color of the dest vertex
function val=getProduct1(A,message,source,dest,color)
    val=1;
    [rows,~]=size(A);
    for k=1:rows
        if A(k,source)~=0 && k~=dest
          val=val*message(k,source,color);  
        end
    end
end

%A-adjacency matrix
%message-message matrix
%dest - the vertex to which the messages are to be multiplied
%color - the color of the dest vertex
function val=getProduct2(A,message,dest,color)
val=1;
[n,~]=size(A);
    for i=1:n
        if A(i,dest)~=0
            val=val*message(i,dest,color);
        end
    end
end