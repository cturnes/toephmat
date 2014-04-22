function T = tltoep(a)
% TLTOEP  Two-level Toeplitz matrix
%
%   T = TLTOEP(A) will return an MN x MN two-level Toeplitz matrix having
%   an N x N pattern of M x M blocks and whose coefficients are drawn from
%   the (2M-1) x (2N-1) coefficient matrix A.
%
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  1.0.0
%   Date:     6-Dec-2013
%
%

    [M, N] = size(a);
    if ((mod(M, 2) == 0) || (mod(N, 2) == 0))
        error('MATLAB:tltoep:badDimensions', ...
             'Coefficient matrix should have odd lengths for dimensions.');
    end
    M = (M+1) / 2;
    N = (N+1) / 2;
    
    % allocate for matrix
    T = zeros(M*N);

    % work on block level if block pattern is smaller
    if (N <= M)

        % main block diagonal
        A0 = toep(a(:, N));
        idx = (1:M);
        for i = 1:N
            T(idx + (i-1)*M, idx + (i-1)*M) = A0;
        end

        % other block diagonals
        for k = 1:(N-1)

            Am = toep(a(:, N-k));
            Ap = toep(a(:, N+k));
            for i = 1:(N-k)
                T(idx + (i-1)*M, idx + (i+k-1)*M) = Am;
                T(idx + (i+k-1)*M, idx + (i-1)*M) = Ap;
            end

        end

    % otherwise work with permuted form
    else

        % (0,0) coords
        A0 = toep(a(M, :));
        idx = (0:(N-1))*M + 1;
        for i = 0:(M-1)
            T(idx+i, idx+i) = A0;
        end

        % other coordinates
        for k = 1:(M-1)
            Am = toep(a(M-k, :));
            Ap = toep(a(M+k, :));
            for i = 0:(M-k-1)
                T(idx + i, idx + i + k) = Am;
                T(idx + i + k, idx + i) = Ap;
            end
        end

    end
           
end