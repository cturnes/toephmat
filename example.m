%{
    @(#)File:          /example.m
    @(#)Package:       Hierarchical Matrix Inversion
    @(#)Version:       1.0.0
    @(#)Last changed:  23 April 2014
    @(#)Author:        C. Turnes
    @(#)Copyright:     Georgia Institute of Technology
    @(#)Purpose:       Example usage - solving a non-uniform FFT system
%}

%% trial parameters

% problem sizes - matrix will be n x n
if (~exist('n', 'var'))
    n = 4096;
end
nFreq = round(3*n);

% number of columns to check for residual
if (~exist('nRes', 'var'))
    nRes = 10;
end

%% build toeplitz matrix coefficients
% choose random frequencies in [-0.5, 0.5)
fk = sort((rand(nFreq, 1) - 0.5));

% build toeplitz coefficients
a = zeros(2*n-1, 1);
q = exp(1j*2*pi*fk);
for k = (1-n):(n-1)
    a(k+n) = sum(q.^k);
end

%% solve a basic system
x = randn(n, 1);
y = tmult(a, x);
% record inverse matrix as well
tic;
[xh, Hi] = tsolve_hier(a, y, 64);
tinit = toc;

%% output information about initial recovery
fprintf('Matrix dimension: %d x %d\n', n, n);
fprintf('\t Initial recovery time:     %4.2e s\n', tinit);
fprintf('\t Residual of solution:      %4.2e\n', norm(xh - x));
fprintf('\t Inverse compression:       %4.2f\n', length(Hi.data) / (n^2));

%% solve new systems
tic;
x = randn(n, nRes);
y = tmult(a, x);
res = zeros(nRes, 1);
for k = 1:nRes
    xh = tsolve_hier(a, y(:, k), [], Hi, 'inv');
    res(k) = norm(x(:, k) - xh);
end
trep = toc / nRes;
fprintf('\t Precomputed recovery time: %4.2e s\n', trep);
fprintf('\t Avg. residual:             %4.2e\n', 10^(mean(log10(res))));