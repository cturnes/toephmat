function [xh, Hi] = tsolve_hier(a, y, nlim, H, spec)
% TSOLVE_HIER  Solve a Toeplitz system through hierarchical matrix
%              inversion
%
%   [XH,HI]=TSOLVE_HIER(A,Y) will return the solution T^{-1}*Y, where T is
%   the Toeplitz matrix constructed from the coefficient array A, as well
%   as the hierarchical matrix representing T^{-1}.
%
%   TSOLVE_HIER(A,Y,NLIM) allows the user to specify the smallest block
%   size in the hierarchical construction.
%
%   TSOLVE_HIER(A,Y,[],H) allows the user to provide the hierarchical
%   representation of T used in calculating its inverse.
%
%   TSOLVE_HIER(A,Y,[],H,SPEC) allows the user to specify whether the
%   provided hierarchical representation H is the representation of the
%   matrix itself (SPEC='mat') or of the Toeplitz inverse (SPEC='inv').
%
%       See also toep toeptrans2hmat hmtimes hminv
%
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  1.0.0
%   Date:     22-Apr-2013
%
%

    % input validation
    if (isempty(a))
        error('MATLAB:tsolve_hier:emptyInput', ...
            'Input coefficient array must be non-empty.');
    end
    if (mod(numel(a), 2) == 0)
        error('MATLAB:tsolve_hier:badInput', ...
            'Input coefficient array must have odd length.');
    end
    if ((nargin < 3) || isempty(nlim))
        nlim = 64;
    end
    
    % get size
    n = (size(a, 1) + 1) / 2;
    if (size(y, 1) ~= n)
        error('MATLAB:tsolve_hier:dimMismatch', ...
            'Input vector y does not agree with dimension of the matrix.');
    end
    
    % build hierarchical structure if necessary
    if (nargin < 4)
        H = toeptrans2hmat(a, nlim);
        Hi = hminv(H);
    else
        if (nargin < 5)
            spec = 'mat';
        end
        switch lower(spec)
            case 'mat'
                Hi = hminv(H);
            case 'inv'
                Hi = H;
            otherwise
                error('MATLAB:tsolve_hier:badSpec', ...
                    'String specifier for hierarchical matrix type is invalid.');
        end
    end
    
    % solve system
    y = sqrt(n)*ifft(spdiags(exp(1j*pi*(0:(n-1))'/n), 0, n, n)*y);
    xh = fft(hmtimes(Hi, y)) / sqrt(n);
    
end