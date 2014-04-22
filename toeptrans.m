function L = toeptrans(a)
% TOEPTRANS  Transformation of a Toeplitz matrix into a corresponding 
%            Loewner form
%
%   T = TOEPTRANS(A) will return an N x N Loewner matrix whose coefficients 
%   are determined by Fourier-like transformations of the (2N-1) x 1
%   coefficient matrix A.
%
%       See also toep
%
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  1.0.0
%   Date:     22-Apr-2013
%
%

    if (isempty(a))
        error('MATLAB:tltoeptrans:emptyInput', ...
            'Input coefficient array must be non-empty.');
    end
    if (mod(numel(a), 2) == 0)
        error('MATLAB:tltoeptrans:badInput', ...
            'Input coefficient array must have odd length.');
    end
    
    % get size
    n = (size(a, 1) + 1) / 2;
    
    % create generator arrays
    u = zeros(n, 2);
    u(1, 2) = -1;
    u(:, 1) = a(n:end) - [0; a(1:(n-1))];
    u = ifft(diag(exp(1j*pi/n*(0:(n-1))')) * u) * sqrt(n);
    v = zeros(n, 2);
    v(n, 1) = -1;
    v(:, 2) = flipud(conj(a(n:end) + [0; a(1:(n-1))]));
    v = ifft(v) * sqrt(n);
    
    % numerator
    Num = u*v';
    
    % denominator
    omplus = exp(1j * 2 * pi / n * (0:(n-1))');
    omminus = omplus * exp(1j*pi/n);
    on = ones(n, 1);
    Denom = (omminus * on' - on * omplus.');
    
    % matrix
    L = Num ./ Denom;

end