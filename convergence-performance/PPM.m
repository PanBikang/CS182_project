

function [x, iter, dist_iter] = PPM(A, Q, xt, opts)

        %%  PPM for the MLE
        % --- INPUT ---
        % A: adjacency matrix (a sparse 0-1 matrix). 
        % x: starting point
        % xt: ground truth
        % opts.tol: desired tolerance for suboptimality
        % opt.T: maximal number of iterations
        % opt.report_interval (optional): number of iterations before we report the progress (default: 10)   
        % opt.print (optional): print iterates or not.
        
        % --- OUTPUT ---
        % x: returned clusters by GPM
        % iter: terminal number of iteration
        % fval_collector: function value of each iteration
        % dist_iter: gap between the iterate and ground truth at each iteration
        fprintf(' ******************** Projected Power Method *************************** \n')
        
       %% Parameter setting
        maxiter = opts.T; 
        tol = opts.tol; 
        if isfield(opts,'report_interval')
            report_interval = opts.report_interval;
        else
            report_interval = 10;  
        end
        if isfield(opts,'print')
            print = opts.print;
        else
            print = 0;
        end
        n = size(A, 1);
        y = Q(:,2) - ones(n,1)'*Q(:,2)*ones(n,1)/n; x = sqrt(n)*y/norm(y);
        Ax = sqrt(n)*A*x; 
        dist_iter(1) = sqrt(2*n^2-2*(x'*xt)^2); %%% compute distance to ground truth || x*x^T - xt*xt^T||_F
        
        for iter = 1:maxiter
                 
               %% Check fixed-point condition
                x1 = x + Ax; [~, inx] = sort(x1); x1 = ones(n,1);
                inxp = inx(1:n/2); x1(inxp) = -1; 
                dist = norm(x - x1);

               %% update of OI + PPM               
                if iter <= floor(log(n)/log(log(n)))
               %% orthogonal iteration with Ritz acceleration
                       [Q, ~] = qr(A*Q, 0);  %% qr decomposition
                       [U, ~] = eig(Q'*A*Q); %% Ritz acceleration
                       Q = Q*U;
                       y = Q(:,2) - ones(n,1)'*Q(:,2)*ones(n,1)/n;
                       x = sqrt(n) * y/norm(y);
                else     
               %% projected power iteration
                       [~, inx] = sort(Ax); x = ones(n,1); 
                       inxp = inx(1:n/2); x(inxp) = -1;                       
                end
                
                Ax = A*x;
                
               %% print and record information
                if mod(iter, report_interval) == 0 && print == 1            
                    fprintf('iternum: %2d, suboptimality: %8.4e \n', iter, dist) 
                end
                               
                dist_iter(iter+1) = sqrt(2*n^2 - 2*(x'*xt)^2);
                
               %%  stopping criterion
                if dist <= tol
                        break;
                end                
                
        end
        

end