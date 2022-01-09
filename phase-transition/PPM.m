

function [x, iter] = PPM(A, Q, opts)

       %%  PPM solves MLE
        % --- INPUT ---
        % A: adjacency matrix (a sparse 0-1 matrix). 
        % x: starting point
        % xt: ground truth
        % opts.tol: desired tolerance for suboptimality
        % opt.T: maximal number of iterations
        % opt.report_interval (optional): number of iterations before we report the progress (default: 100)   
        % opt.print (optional): print iterates or not.
        
        % --- OUTPUT ---
        % x:    returned clusters by GPM
        % iter:  terminal number of iteration
        % val_collector: function value of each iteration
                
       %% Parameter setting
        maxiter = opts.T; 
        tol = opts.tol; 
        if isfield(opts,'report_interval') %% whether "report_interval" is inside opts
            report_interval = opts.report_interval;
        else
            report_interval = 100;  
        end
        if isfield(opts,'print')
            print = opts.print;
        else
            print = 1;
        end
        n = size(A, 1);
        y = Q(:,2); x = sqrt(n)*y/norm(y);  %% initialize       
        Ax = A*x; 
        
        for iter = 1:maxiter                               

               x_old = x;

               %% update of OI + PPM               
                if iter <= floor(0.2*log(n)/log(log(n)))
               %% orthogonal iteration with Ritz acceleration
                       [Q, ~] = qr(A*Q, 0); %% qr decomposition
                       [U, ~] = eig(Q'*A*Q); %% Ritz acceleration
                       Q = Q*U;
                       y = Q(:,2) - ones(n,1)'*Q(:,2)*ones(n,1)/n;
                       x = sqrt(n) * y/norm(y);
                else     
               %% projected power iteration
                       [~, inx] = sort(Ax); x = ones(n,1);  %% inx is the index of sorted Ax
                       inxp = inx(1:n/2); x(inxp) = -1;     %% x(inxp == 0) = -1                  
                end
                
                Ax = A*x;   %% iterate
                
               %% print and record information                                              
                if mod(iter, report_interval) == 0 && print == 1            
                    fprintf('iternum: %2d, suboptimality: %8.4e\n', iter, dist) 
                end                             
                
               %%  stopping criterion             
                if norm(x-x_old) <= tol
                    break;
                end
        end
        

end