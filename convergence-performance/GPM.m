

function [x, iter, dist_iter] = GPM(A, Q, xt, opts)

        %%  GPM for the regularized MLE
        % --- INPUT ---
        % A: adjacency matrix (a sparse 0-1 matrix). 
        % x: starting point
        % xt: ground truth
        % opts.rho: regularizer parameter
        % opts.tol: desired tolerance for suboptimality
        % opt.T: maximal number of iterations
        % opt.report_interval (optional): number of iterations before we report the progress (default: 10)   
        % opt.quiet (optional): print iterates or not.
        
        % --- OUTPUT ---
        % x:    returned clusters by GPM
        % iter:  terminal number of iteration
        % fval_collector: function value of each iteration
        % dist_iter: gap between the iterate and ground truth at each iteration
                
       %% Parameter setting
        rho = opts.rho;  
        maxiter = opts.T; 
        tol = opts.tol; 
        if isfield(opts,'report_interval')
            report_interval = opts.report_interval;
        else
            report_interval = 10;  
        end
        if isfield(opts,'quiet')
            quiet = opts.quiet;
        else
            quiet = false;
        end
        n = size(A, 1); x = sqrt(n)*Q(:,2);
        Ax = A*x;
        dist_iter(1) = sqrt(2*n^2 - 2*(x'*xt)^2); %%% compute distance to ground truth || x*x^T - xt*xt^T||_F
        fprintf(' ******************** Generalized Power Method *************************** \n')
        
        for iter = 1:maxiter
                 
                xold = x;
                
               %% Check fixed-point condition
                x1 = x + Ax - rho*ones(n,1)'*x*ones(n,1);  x1(x1>=0) = 1; x1(x1<0) = -1; 
                dist = norm(x - x1);
                
               %% update of GPM + PGD                
                if iter <= floor(log(n)/log(log(n)))
               %% power iteration
                       x = Ax - rho*ones(n,1)'*x*ones(n,1); x = x/norm(x)*sqrt(n);      
                else  
               %% generalized power iteration     
                       x = Ax - rho*ones(n,1)'*x*ones(n,1);
                       x( x >= 0) = 1;                                      
                       x( x < 0) = -1;
                end
                
                Ax = A*x;             
                
               %% print and record information
                if mod(iter, report_interval) == 0 && ~quiet            
                    fprintf('iternum: %2d, suboptimality: %8.4e\n', iter, dist) 
                end
                               
                dist_iter(iter+1) = sqrt(2*n^2 - 2*(x'*xt)^2);
                
               %%  stopping criterion
                if dist <= tol && norm(x-xold) == 0 
                        break;
                end                
                
        end
        

end