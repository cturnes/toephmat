function y = tmult(a, x)
% TMULT  Toeplitz matrix multiplication
%
%   Y = TMULT(A, X) will return the multiplication of the M x N Toeplitz
%   matrix T, whose coefficients are defined by the (M+N-1) x 1 vector A,
%   with the N x 1 vector X.
%
%
%       See also toep
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  2.0.0
%   Date:     22-Apr-2014
%
%

    % check for valid inputs
    if (nargin < 2)
        error('MATLAB:minrhs','Not enough input arguments.');
    end

    % check sizes
    n = size(x, 1);
    m = length(a) - n + 1;
    if (m < 0)
        error('MATLAB:tmult:innerDimMismatch', ...
            'Inner dimmensions do not match');
    end
    
    % start convolution
    A = fft(a, m+2*n -2);
    
    % perform convolution
    y = ifft(fft(x, length(A)) .* A(:, ones(1, size(x, 2))));
    
    % truncate
    y = y(n:(n+m-1), :);

end
