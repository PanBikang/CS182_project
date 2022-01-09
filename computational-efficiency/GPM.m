

function [x, iter, fval_collector] = GPM(A, Q, opts)

        %%  GPM for the regularized MLE
        % --- INPUT ---
        % A: adjacency matrix (a sparse 0-1 matrix). 
        % x: starting point
        % xt: ground truth
        % opts.rho: regularizer parameter
        % opts.tol: desired tolerance for suboptimality
        % opts.T: maximal number of iterations
        % opts.report_interval (optional): number of iterations before we report the progress (default: 10)   
        % opts.quiet (optional): print iterates or not
        % opts.init_iter: parameter controlling iterations number of the 1st stage
        
        % --- OUTPUT ---
        % x:    returned clusters by GPM
        % iter:  terminal number of iteration
        % fval_collector: function value of each iteration
        % dist_iter: gap between the iterate and ground truth at each iteration
                
       %% Parameter setting
        rho = opts.rho;  
        maxiter = opts.T; 
        tol = opts.tol; 
        t = opts.init_iter;
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
        Ax = A*x; fval = -x'*Ax + rho*(ones(n,1)'*x)^2;  %%% compute function value
        fval_collector(1) = fval;  
        % fprintf(' ******************** Generalized Power Method *************************** \n')
        
        for iter = 1:maxiter
                 
               %% Check fixed-point condition
                x1 = x + Ax - rho*ones(n,1)'*x*ones(n,1);  
                x1(x1>=0) = 1; x1(x1<0) = -1; 
                dist = norm(x - x1);
                
               %% update of PI + GPI                
                if iter <= floor(t*log(n)/log(log(n)))
               %% power iteration (PI)
                       x = Ax - rho*ones(n,1)'*x*ones(n,1); x = x/norm(x)*sqrt(n);      
                else  
               %% generalized power iteration (GPI)     
                       x = Ax - rho*ones(n,1)'*x*ones(n,1);
                       x(x>=0) = 1;                                      
                       x(x<0) = -1;
                end
                
                Ax = A*x; 
                fval = -x'*Ax + rho*(ones(n,1)'*x)^2;                 
                
               %% print and record information
                if mod(iter, report_interval) == 0 && ~quiet            
                    fprintf('iternum: %2d, suboptimality: %8.4e, fval: %.3f \n', iter, dist, fval) 
                end
                                
                fval_collector(iter+1) = fval; 
                
               %%  stopping criterion
                if dist <= tol
                    break;
                end                
                
        end
        

end