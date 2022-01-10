
clear; clc;

%% Load the Real-World Datase
% load Datasets/polblogs          %%%%%%% load real_datasets\polblogs\polblogs.louvain;
% As = Problem.A; As(As>0) = 1;   %%%%% take the adjacency matrix
% xt = Problem.aux.nodevalue;     %%%%% take the ground truth
% n1 = sum(xt==0); n2 = sum(xt==1); n = n1+n2;
% idx = n1-n2+1:n; As = As(idx, idx);
% xt = xt(n1-n2+1:n); 
% n = size(As,1); K=2; 
% xt = (xt - 0.5)*2;
n = 3000;      %%% n = the number of nodes
K = 2;        %%% K = the number of communities
m = n/K;      %%% m = the community size
nnt = 40;     %%% the number of repeating the trials for fixed alpha, beta
tol = 1e-3;   %%% tolerance of success recovery
run_PPM = 1; run_MGD = 1; run_GPM = 1; run_SDP = 1; run_SC = 1;self_loops = 0;
%% ground truth 
% Xt = kron(eye(K), ones(m));     %% tensor product to produce
% Xt(Xt==0)=-1;                   %% change all 0 to -1
%                                 %%% Xt = the true cluster matrix
xt = [ones(m,1); -ones(m,1)];   %% first 150 first class      
                                %%%  xt = the true cluster vector
% randIndex = randperm(size(xt,1));
% xt = xt(randIndex,:);
Xt = xt .* xt';
mu = 2;
gamma = exp(-1.5);
Xloc = (randn(n, 1) + mu) .* xt;
Xdis = abs(Xloc - Xloc');
Xprob = gamma * exp(-Xdis);
A_rand = rand(n);
A = zeros(n);
for i = 1:n
    for j = 1:n
        if A_rand(i,j) <= Xprob(i,j)
            A(i,j) = 1;
        end
    end
end
if self_loops == 0
    A = A - diag(diag(A));
end
As = sparse(A);
%% initial setting
iternum1 = 10;
[fval_PPM, fval_GPM, fval_MGD, fval_SDP] = deal(zeros(iternum1,1));
[PPM_collector, GPM_collector, MGD_collector, SDP_collector] =  deal(zeros(n,n, iternum1));
[ttime_PPM, ttime_GPM, ttime_MGD, ttime_SDP, ttime_SC] = deal(0);



parfor iter = 1:iternum1
    
        fprintf('Iter Num: %d \n', iter);        
        
        %% generate the initial point
        rng(iter); Q = randn(n,2); Q0 = Q*(Q'*Q)^(-0.5);  
        
        %% set the parameters in the running methods
        maxiter = 2e3; tol = 1e-3; report_interval = 4e3; total_time = 1e3; print = 0; 
        
        %% Manifold Gradient Descent
        if run_MGD == 1
            rho = nnz(As)/(n^2);               
            opts = struct('rho', rho, 'T', maxiter, 'tol', tol, 'report_interval', report_interval, 'print', print);
            tic; [Q, iter_MGD, val_collector_MGD] = manifold_GD(As, Q0, opts); time_MGD=toc;
            ttime_MGD = ttime_MGD + time_MGD;
            X_MGD = Q*Q'; MGD_collector(:,:,iter) = X_MGD;
            fval_MGD(iter) = -trace(Q'*As*Q);
        end
                
        %% Spectral Clustering
        if run_SC == 1
            tic;
            [U,D] = eigs(As+As', 2); ev = diag(D);           
            [~,I] = sort(ev,'descend'); ev = ev(I);          
            U = U(:,I); U = normr(U);
            e_SC = kmeans(U, K, 'replicates', 20);
            time_SC = toc;
            ttime_SC = ttime_SC + time_SC;
        end

        %% PPM for MLE
        if run_PPM == 1
            opts = struct('T', maxiter, 'tol', tol,'report_interval', report_interval,...
                'total_time', total_time, 'init_iter', 1e1, 'print', print);
            tic; [x_PPM, iter_PPM, val_collector_PPM] = PPM(As, Q0, opts); time_PPM=toc;
            ttime_PPM = ttime_PPM + time_PPM;
            X_PPM = x_PPM*x_PPM'; PPM_collector(:,:,iter) = X_PPM;
            fval_PPM(iter) = -x_PPM'*As*x_PPM;
        end
        
        %% GPM for Regularized MLE
        if run_GPM == 1
            rho = 1e-3; % rho = sum(sum(As))/n^2;            
            opts = struct('T', maxiter, 'rho', rho, 'tol', tol, 'report_interval', report_interval,...
                'total_time', total_time, 'init_iter', 1e1, 'print', print); 
            tic; [x_GPM, iter_GPM, fval_collector_GPM] = GPM(As, Q0, opts); time_GPM=toc; 
            ttime_GPM = ttime_GPM + time_GPM;
            X_GPM = x_GPM*x_GPM'; GPM_collector(:,:,iter) = X_GPM;
            fval_GPM(iter) = -x_GPM'*As*x_GPM + rho*sum(x_GPM)^2;
        end
        
        %% Solve the SDP to recover X
        if run_SDP == 1
            X0 = Q0*Q0';
            opts = struct('rho', 10+iter, 'T', maxiter, 'tol', tol, 'quiet', 1, 'report_interval', report_interval);
            tic; X_SDP = sdp_admm1(As, X0, 2, opts); time_SDP=toc;
            ttime_SDP = ttime_SDP + time_SDP;
            SDP_collector(:,:,iter) = X_SDP; fval_SDP(iter) = -trace(X_SDP'*As);
        end  
         
end 
    
%% plot the figures
color_choice = 8;
figure; imagesc(xt*xt'); title('Ground truth');

if run_PPM == 1
    [min_PPM, inx] = min(fval_PPM); e_PPM = labelsFromX(PPM_collector(:,:,inx),K); 
    e_PPM = (e_PPM - 1.5)*2; time_PPM = ttime_PPM/10; 
    dist_PPM = min(nnz(e_PPM-xt), nnz(e_PPM+xt));
    figure; imagesc(e_PPM*e_PPM'); title('PPM'); 
end

if run_GPM == 1
    [min_GPM, inx] = min(fval_GPM); e_GPM = labelsFromX(GPM_collector(:,:,inx),K); 
    e_GPM = (e_GPM - 1.5)*2; time_GPM = ttime_GPM/10; 
    dist_GPM = min(nnz(e_GPM-xt), nnz(e_GPM+xt));
    figure; imagesc(e_GPM*e_GPM'); title('GPM'); 
end

if run_MGD == 1
    [min_MGD, inx] = min(fval_MGD); e_MGD = labelsFromX(MGD_collector(:,:,inx),K); 
    e_MGD = (e_MGD - 1.5)*2; time_MGD = ttime_MGD/10; 
    dist_MGD = min(nnz(e_MGD-xt), nnz(e_MGD+xt));    
    figure; imagesc(e_MGD*e_MGD'); title('MGD'); 
end

if run_SDP == 1
    [min_SDP, inx] = min(fval_SDP); e_SDP = labelsFromX(SDP_collector(:,:,inx),K); 
    e_SDP = (e_SDP - 1.5)*2; time_SDP = ttime_SDP/10; 
    dist_SDP = min(nnz(e_SDP-xt), nnz(e_SDP+xt));    
    figure; imagesc(e_SDP*e_SDP'); title('SDP'); 
end

if run_SC == 1
    time_SC = ttime_SC/10; e_SC = (e_SC - 1.5)*2;
    dist_SC = min(nnz(e_SC-xt), nnz(e_SC+xt));    
    figure; imagesc(e_SC*e_SC'); title('SC'); 
end


    