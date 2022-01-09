

function [x, iter, fval_collector] = PPM(A, Q, opts)

        %%  PPM solves MLE
        % --- INPUT ---
        % A: adjacency matrix (a sparse 0-1 matrix). 
        % x: starting point
        % xt: ground truth
        % opts.tol: desired tolerance for suboptimality
        % opts.T: maximal number of iterations
        % opts.report_interval (optional): number of iterations before we report the progress (default: 100)   
        % opts.print (optional): print iterates or not.
        % opts.init_iter: parameter controlling iterations number of the 1st stage
        
        % --- OUTPUT ---
        % x: returned clusters by GPM
        % iter: terminal number of iteration
        % fval_collector: function value of each iteration
                
       %% Parameter setting
        maxiter = opts.T; 
        tol = opts.tol; 
        t = opts.init_iter;
        if isfield(opts,'report_interval')
            report_interval = opts.report_interval;
        else
            report_interval = 1;  
        end
        if isfield(opts,'print')
            print = opts.print;
        else
            print = 0;
        end
        n = size(A, 1); q = Q(:,2);
        y = q - sum(q)/n; x = sqrt(n)*y/norm(y); 
        Ax = A*x; fval = -x'*Ax;  %%% compute function value
        fval_collector(1) = fval;  
                
        for iter = 1:maxiter
                
               %% Check fixed-point condition
                x1 = x + Ax; [~, inx] = sort(x1); x1 = ones(n,1);
                inxp = inx(1:n/2); x1(inxp) = -1; 
                dist = norm(x - x1);

               %% update of OI + PPI               
                if iter <= floor(t*log(n)/log(log(n)))
               %% orthogonal iteration (OI) with Ritz acceleration    
                       [Q, ~] = qr(A*Q, 0); %% qr decomposition
                       [U, ~] = eigs(Q'*A*Q); %% Ritz acceleration
                       Q = Q*U; q = Q(:,2);
                       y = q - sum(q)/n;
                       x = sqrt(n)*y/norm(y);
                else     
               %% projected power iteration (PPI)
                       [~, inx] = sort(Ax); x = ones(n,1);
                       x(inx(1:n/2)) = -1;   
                end
                
                Ax = A*x; fval = -x'*Ax; 
               
               %% prinf and record the information 
                if mod(iter, report_interval) == 0 && print == 1            
                    fprintf('iternum: %2d, suboptimality: %8.4e, fval: %.3f \n', iter, dist, fval) 
                end
                                
                fval_collector(iter+1) = fval; 

               %%  stopping criterion
                if dist <= tol
                    break;
                end                
                
        end
        

end