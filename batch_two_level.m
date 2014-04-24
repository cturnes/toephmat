%{
    @(#)File:          /batch_two_level.m
    @(#)Package:       Hierarchical Matrix Inversion
    @(#)Version:       1.0.0
    @(#)Last changed:  24 April 2014
    @(#)Author:        C. Turnes
    @(#)Copyright:     Georgia Institute of Technology
    @(#)Purpose:       Batch two-level Toeplitz inversion experiments
%}

%% parameters and setup
clear; clc;

% range of sizes
nrange = (2.^(0:12)).';
 
% compression ratios:
comp_ratios = NaN(length(nrange), length(nrange), 2);

%% conduct experiment
for i = 1:length(nrange)
    
    m = nrange(i);
    
    for j = 1:(length(nrange) + 1 - i)
    
        n = nrange(j);
        
        % build toeplitz coefficient
        a = randn(2*m-1, 2*n-1);
        
        % slow, but no direct hierarchical constructor yet...
        L = tltoeptrans(a);
        P = sparse(morton2(m, n), (1:(m*n)), 1, m*n, m*n);
        L = P'*L*P;
        H = dense2hm(L);
        clear L;
        
        % compute compression amount
        comp_ratios(i, j, 1) = length(H.data) / (m^2*n^2);
        
        % invert
        Hi = hminv(H);
        comp_ratios(i, j, 2) = length(Hi.data) / (m^2*n^2);
        
        fprintf('Completed m = %d, n = %d, compression ratios = %4.2f and %4.2f\n', m, n, comp_ratios(i, j, 1), comp_ratios(i, j, 2));
    
    end
    
end