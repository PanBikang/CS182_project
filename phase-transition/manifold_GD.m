

function [Q, iter] = manifold_GD(A, Q, opts)

        %%  MGD solves the MFO in Bandeira et al. (2016)
        % --- INPUT ---
        % A: adjacency matrix (a sparse 0-1 matrix). 
        % Q: starting point
        % xt: ground truth
        % opts.rho: regularizer parameter in MFO
        % opts.tol: desired tolerance for suboptimality
        % opt.T: maximal number of iterations
        % opt.report_interval (optional): number of iterations before we report the progress (default: 100)   
        % opt.print (optional): print iterates or not.
        
        % --- OUTPUT ---
        % Q: returned clusters by GPM
        % iter: terminal number of iteration
        % fval_collector: function value of each iteration

       %% Parameter setting
        rho = opts.rho;     %%%  regularizer parameter 
        maxiter = opts.T;  %%% maximal iteration number
        tol = opts.tol;        %%% tolerance
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
        n = size(A, 1); 
        eta = 2; sigma = 0.4; %%% line search parameters        
        fnew = -trace(Q'*A*Q) + rho*norm(Q'*ones(n,1))^2; %%% compute function value
        fval_collector(1) = fnew; 
        step_init = 1; Qnew = Q; 
        
        for iter = 1:maxiter
                   
                Q = Qnew; fval=fnew;
                
               %% manifold gradient descent with line search
                stepsize = step_init; 
                x = -sum((A*Q).*Q,2) + rho*Q*(Q'*ones(n,1));
                projgrad = -A*Q + rho*repmat(sum(Q),n,1) - repmat(x,1,2).*Q; 
                Qnew = Q - stepsize * projgrad;
                Qnew = normr(Qnew);
                
                fnew = -trace(Qnew'*A*Qnew) + rho*norm(Qnew'*ones(n,1))^2;
                count = 0;
                
                while fnew > fval - sigma/2*1/stepsize*norm(Qnew - Q)^2
                      stepsize = stepsize/eta;
                      Qnew = Q - stepsize * projgrad;
                      Qnew = normr(Qnew);
                      fnew = -trace(Qnew'*A*Qnew) + rho*norm(Qnew'*ones(n,1))^2;
                      count = count + 1;
                      if count >= 20
                          break;
                      end
                end
          
               %% print and record information 
                if mod(iter,report_interval) == 0 && print == 1            
                    fprintf('iternum: %2d, grad_norm: %8.4e, stepsize: %.4f \n', iter, norm(projgrad), stepsize) 
                end
                
               %%   stopping criterion
                if norm(projgrad) <= tol 
                        break;
                end                

        end
        

end