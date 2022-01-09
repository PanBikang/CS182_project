
function x = SC(A)

    %% Eigs in matlab for vanilla spectral clustering in Abbe et al (2017)    
    % --- INPUT ---
    % A:   Adjacency matrix (a sparse 0-1 matrix). 

    % --- OUTPUT ---
    % x:   cluster matrix
    
    [U,~] = eigs(A, 2);
    x = U(:,2);
    x(x>=0) = 1;
    x(x<0) = -1;
    
end