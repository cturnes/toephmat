function T = toep(a, n)
% TOEP  Toeplitz matrix
%
%   T = TOEP(A) will return an N x N Toeplitz matrix whose coefficients are
%   obtained from the 2N-1 x 1 vector A.  If the length of A is not odd, an
%   error message will be displayed.  A(1) will correspond to the northeast
%   corner of T, A(N) to the northwest corner of T, and A(end) to the
%   southwest corner of T.
%
%   T = TOEP(A, N) will perform the same, but allow the user to specify
%   the matrix T as being (LENGTH(A)-N+1) x N.  In this case, the only
%   requirement is that LENGTH(A) >= N.
%
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  1.0.0
%   Date:     21-Mar-2013
%
%

    a = a(:);
    if (nargin < 2)
        Ntot = length(a);
        if (mod(Ntot, 2) == 0)
            error('If you do not supply a size vector, then a should have odd length');
        end
        n = (Ntot+1) / 2 * ones(2, 1);
    else
        if (length(a) < n)
            error('Input vector is not long enough to construct matrix with supplied number of columns');
        end
    end
    
    T = toeplitz(a(n(2):end), a(n(2):-1:1));

end