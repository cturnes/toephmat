%{
    @(#)File:          /batch_scalar.m
    @(#)Package:       Hierarchical Matrix Inversion
    @(#)Version:       1.0.0
    @(#)Last changed:  24 April 2014
    @(#)Author:        C. Turnes
    @(#)Copyright:     Georgia Institute of Technology
    @(#)Purpose:       Batch scalar Toeplitz inversion experiments
%}

%% parameters and setup
clear; clc;

% range of sizes
NN = 2.^(6:13);
 
% number of experiments per matrix size
nExp = 10;

% number of columns to use to average execution times
nRes = 100;

% allocate for results
init_times = zeros(nExp, length(NN));
comp_rates = zeros(nExp, length(NN));
rept_times = zeros(nExp, length(NN));
mean_resid = zeros(nExp, length(NN));

%% conduct experiment
for m = 1:length(NN)
    
    % select size
    n = NN(m);
    
    % conduct experiments
    for r = 1:nExp
        
        example;
        % record data
        init_times(r, m) = tinit;
        rept_times(r, m) = trep;
        comp_rates(r, m) = length(Hi.data) / (n^2);
        mean_resid(r, m) = 10^(mean(log10(res)));
        
        % clear out trial data
        clear fk q a xh Hi y x res tinit trep;
    end
    
end