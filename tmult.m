function y = tmult(a, x, p)
% TMULT  Toeplitz matrix multiplication
%
%   Y = TMULT(A, X) will return the multiplication of the M x N Toeplitz
%   matrix T, whose coefficients are defined by the (M+N-1) x 1 vector A,
%   with the N x 1 vector X.
%
%   Y = TMULT(A, X, P) will do the same, using a maximum FFT prime factor
%   of P (default = inf).
%
%       See also toep
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  2.0.0
%   Date:     1-Oct-2013
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
        error('MATLAB:tmultfcn:innerDimMismatch', ...
            'Inner dimmensions do not match');
    end
    
    % check if min prime factor is supplied, start convolution
    if (nargin < 3)
        A = cfft(a, m+2*n -2);
    else
        
        if (isempty(p))
            warning('MATLAB:tmult:noPrimeBound', ...
                'Third argument is empty; setting default prime bound to 5');
            p = 5;
        else
            if (length(p) > 1)
                warning('MATLAB:tmult:primeBoundNotScalar', ...
                    'Prime bound must be integer scalar greater than 1.');
                p = p(1);
            end
            if (p < 2)
                error('MATLAB:tmult:primeBoundTooSmall', ...
                    'Prime bound must be integer scalar greater than 1.');
            end
        end
        A = cfft(a, m+2*n-2, p);
    end
    
    % perform convolution
    y = ifft(cfft(x, length(A)) .* A(:, ones(1, size(x, 2))));
    
    % truncate
    y = y(n:(n+m-1), :);

end