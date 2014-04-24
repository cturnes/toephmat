function [U,S,V] = lrsvd(A, r, rk)
% LRSVD  Algorithm to compute the SVD of a low-rank matrix
%
%   [U,S,V] = LRSVD(A) will look to reduce the computation involved in
%   computing the SVD of a low-rank matrix through the use of random
%   matrices.  The algorithm will multiply the matrix A by an R x SIZE(A,1)
%   on the left and SIZE(A,2) x R matrix on the right, where R <
%   0.5*MIN(SIZE(A)).  If the resulting matrix is full-rank, R will be
%   doubled; if not, the rank of the resulting matrix is the rank of A, and
%   a more efficient SVD can be computed.  If R increases beyond its
%   maximum limit, the regular SVD function will be called.
%
%   [U,S,V] = LRSVD(A,RMAX) allows the user to specify the upper bound of R
%   as RMAX*MIN(SIZE(A)).
%
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  1.0.0
%   Date:     22-Apr-2013
%
%

    % ratio requirement
    if ((nargin < 2) || (isempty(r)))
        r = 0.5;
    end
    
    if (nargin < 3)
    
        % try to determine rank
        i = 64;
        while (i <= round(r*min(size(A))))
            
            % random combination
            BL = randn(i, size(A, 1));
            BR = randn(size(A, 2), i);
            q = BL*A*BR;
            rk = rank(q);
            if (rk < i)
                break;
            end
            i = i * 2;
            
        end
    else
        i = 0;
        BL = randn(rk, size(A, 1));
        BR = randn(size(A, 2), rk);
    end
    
    % fall back to regular svd
    if (i > round(r*min(size(A))))
        
        [U,S,V] = svd(A);
        
    % efficient approximate SVD computation
    else
       [Q1,R1] = qr(A*BR(:, 1:rk),0); %#ok<NASGU>
       [Q2,R2] = qr((BL(1:rk, :)*A)', 0); %#ok<NASGU>
       Q1 = Q1(:, 1:rk);
       Q2 = Q2(:, 1:rk);
       B = Q1'*A*Q2;
       [U,S,V] = svd(B);
       U = Q1*U;
       V = Q2*V;
    end

end