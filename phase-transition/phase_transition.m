
clearvars; clc;

%% basic setting
n = 300;      %%% n = the number of nodes
K = 2;        %%% K = the number of communities
m = n/K;      %%% m = the community size
nnt = 40;     %%% the number of repeating the trials for fixed alpha, beta

%% ground truth 
Xt = kron(eye(K), ones(m)); 
Xt(Xt==0)=-1;                           %%% Xt = the true cluster matrix
xt = [ones(m,1); -ones(m,1)];           %%%  xt = the true cluster vector

%% set the ranges of alpha, beta
arange = 0:0.5:30; brange = 0:0.4:10; arange(1) = 0.01;
nna = length(arange); nnb = length(brange);

%% record information
[prob_SDP, prob_MGD, prob_SC, prob_GPM, prob_PPM]  = deal(zeros(nna,nnb));  %%% record ratio of exact recovery
[ttime_PPM, ttime_MGD, ttime_SC, ttime_GPM, ttime_SDP] = deal(0);  %%% record total running time

%% choose the running algorithm
run_SDP = 1; run_MGD = 1; run_SC = 1; run_GPM = 1; run_PPM = 1;

%% with and without self-loops
self_loops = 0; %%% 1 = self-loops; 0 = no self-loops

for iter1 = 1:nna      %%% choose alpha
    
    a=arange(iter1); 
    
    for iter2 = 1:nnb  %%%  choose beta 
            
        b = brange(iter2); 
        p = a*log(n)/n; q=b*log(n)/n; %%% p: the inner connecting probability; q: the outer connecting probability;           
        [succ_FW, succ_SDP, succ_MGD, succ_SC, succ_GPM, succ_PPM] = deal(0);

        for iter3 = 1:nnt %%% the number of repeating the trials 

                %% generate an adjacency matrix A by Binary SBM
                Ans11 = rand(m); Al11 = tril(Ans11,-1);                      
                As11 = Al11 + Al11'+diag(diag(Ans11));
                A11 = double(As11<=p);
                As12 = rand(m); A12 = double(As12<=q);
                Ans22 = rand(m); Al22 = tril(Ans22,-1);                    
                As22 = Al22 + Al22' + diag(diag(Ans22));
                A22 = double(As22<=p);
                A = ([A11,A12;A12',A22]); 
                if self_loops == 0
                    A = A - diag(diag(A));
                end
                A = sparse(A);
                                
                %% choose the initial point
                Q = randn(n,2); Q0 = Q*(Q'*Q)^(-0.5);
                
                %% set the parameters in the running methods
                maxiter = 50; tol = 1e-3; report_interval = 1e2; total_time = 1e3;
                
                %% PPM for MLE
                if run_PPM == 1
                        opts = struct('T', 20, 'tol', tol, 'report_interval', report_interval, 'total_time', total_time);
                        tic; [x_PPM, iter_PPM] = PPM(A, Q0, opts); time_PPM=toc;
                        ttime_PPM = ttime_PPM + time_PPM;
                        dist_PPM =  min(norm(x_PPM-xt), norm(x_PPM+xt));
                        if dist_PPM <= 1e-3
                                succ_PPM = succ_PPM + 1;
                        end
                end
                
                %% GPM for regularized MLE
                if run_GPM == 1
                        opts = struct('T', 20, 'rho', sum(sum(A))/n^2, 'tol', tol, 'report_interval', report_interval, 'total_time', total_time);
                        tic; [x_GPM, iter_GPM] = GPM(A, Q0, opts); time_GPM=toc;
                        ttime_GPM = ttime_GPM + time_GPM;
                        dist_GPM =  min(norm(x_GPM-xt), norm(x_GPM+xt));
                        if dist_GPM <= 1e-3
                                succ_GPM = succ_GPM + 1;
                        end
                end
                
                %% Manifold Gradient Descent (MGD)
                if run_MGD == 1
                        opts = struct('rho', (p+q)/2, 'T', maxiter, 'tol', tol,'report_interval', report_interval, 'total_time', total_time);                
                        tic; [Q, iter_MGD] = manifold_GD(A, Q0, opts); time_MGD=toc;
                        ttime_MGD = ttime_MGD + time_MGD;
                        X_MGD = Q*Q';
                        dist_MGD =  norm(X_MGD-Xt, 'fro');
                        if dist_MGD <= 1e-3
                                succ_MGD = succ_MGD + 1;
                        end
                end

                %% ADMM for SDP
                if run_SDP == 1
                        X0 = Q0*Q0';
                        opts = struct('rho', 1, 'T', maxiter, 'tol', 1e-1, 'quiet', true, ...
                                'report_interval', report_interval, 'total_time', total_time);
                        tic; X_SDP = sdp_admm1(A, Xt, X0, 2, opts); time_SDP = toc;
                        ttime_SDP = ttime_SDP + time_SDP;
                        Ht = [ones(m,1) zeros(m,1); zeros(m,1) ones(m,1)]; 
                        X_SDP(X_SDP >= 0.5) = 1; X_SDP(X_SDP < 0.5) = 0;
                        dist_SDP =  sqrt(n^2 - 2*trace(Ht'*X_SDP*Ht));
                        if dist_SDP <= 1e-3
                                succ_SDP = succ_SDP + 1;
                        end
                end

                %% Spectral clustering
                if run_SC == 1
                    tic; x_SC = SC(A); time_SC = toc;
                    ttime_SC = ttime_SC + time_SC;
                    dist_SC =  min(norm(x_SC-xt), norm(x_SC+xt));
                    if dist_SC <= 1e-3
                                succ_SC = succ_SC + 1;
                    end
                 end

                fprintf('Outer iter: %d, Inner iter: %d,  Repated Num: %d \n', iter1, iter2, iter3);
        end

        prob_PPM(iter1, iter2) = succ_PPM/nnt;
        prob_GPM(iter1, iter2) = succ_GPM/nnt;
        prob_SDP(iter1, iter2) = succ_SDP/nnt;
        prob_MGD(iter1, iter2) = succ_MGD/nnt;
        prob_SC(iter1, iter2) = succ_SC/nnt;
            
    end    
end 

%% Plot the figures of phase transition
f =  @(x,y)  sqrt(y) - sqrt(x) - sqrt(2); 

if run_PPM == 1
    figure(); imshow(prob_PPM, 'InitialMagnification','fit','XData',[0 10],'YData',[0 30]); colorbar; 
    axis on; set(gca,'YDir','normal'); hold on; 
    fimplicit(f,[0 10 0 30], 'LineWidth', 1.5, 'color', 'r');
    daspect([1 3 1]);
    set(gca,'FontSize', 12, 'FontWeight','bold');
    xlabel('\beta', 'FontSize', 16); ylabel('\alpha', 'FontSize', 16); title('PPM');
end

if run_GPM == 1
    figure(); imshow(prob_GPM, 'InitialMagnification','fit','XData',[0 10],'YData',[0 30]); colorbar; 
    axis on; set(gca,'YDir','normal'); hold on; 
    fimplicit(f,[0 10 0 30], 'LineWidth', 1.5, 'color', 'r');
    daspect([1 3 1]);
    set(gca,'FontSize', 12, 'FontWeight','bold');
    xlabel('\beta', 'FontSize', 16); ylabel('\alpha', 'FontSize', 16); title('GPM');
end

if run_SC == 1
    figure();
    imshow(prob_SC, 'InitialMagnification','fit','XData',[0 10],'YData',[0 30]); colorbar; 
    axis on; set(gca,'YDir','normal'); hold on; 
    fimplicit(f,[0 10 0 30], 'LineWidth', 1.5, 'color', 'r');
    daspect([1 3 1]);
    set(gca,'FontSize', 12, 'FontWeight','bold');
    xlabel('\beta', 'FontSize', 16); ylabel('\alpha', 'FontSize', 16); title('SC');
end

if run_SDP == 1
    figure();
    imshow(prob_SDP, 'InitialMagnification','fit', 'XData',[0 10],'YData',[0 30]); colorbar;
    axis on; set(gca,'YDir','normal'); hold on; 
    fimplicit(f,[0 10 0 30], 'LineWidth', 1.5, 'color', 'r');
    daspect([1 3 1]);
    set(gca,'FontSize', 12, 'FontWeight','bold');
    xlabel('\beta', 'FontSize', 16); ylabel('\alpha', 'FontSize', 16); title('SDP');
end

if run_MGD == 1
    figure(); 
    imshow(prob_MGD, 'InitialMagnification','fit','XData',[0 10],'YData',[0 30]); colorbar; 
    axis on; set(gca,'YDir','normal'); hold on; 
    fimplicit(f,[0 10 0 30], 'LineWidth', 1.5, 'color', 'r'); daspect([1 3 1]);
    set(gca,'FontSize', 12, 'FontWeight','bold');
    xlabel('\beta', 'FontSize', 16); ylabel('\alpha', 'FontSize', 16); title('MGD');
end


