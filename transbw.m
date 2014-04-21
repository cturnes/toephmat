function B = transbw(A, m)
% TRANSBW  Blockwise transpose
%
%   B = TRANSBW(A, M) will return a matrix whose individual blocks have  
%   been transposed.  The matrix A is defined to have M(1) x M(2) 
%   blocks.  If M only has one entry, it is assumed that M(2) = M(1).
%
%       See also transbl ptransbl ptransbw
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  2.0.0
%   Date: 12-Dec-2013
%
%

    if (nargin < 2)
        warning('BlockwiseOp:No_Block_Size', ...
            'No supplemental arguments - assuming block pattern is 1 x 1');
        B = A';
        return;
    end
    if (length(m) == 1)
        m(2) = m(1);
    end

    % find block pattern size
    [n1, n2] = size(A);
    n1 = n1 / m(1);
    n2 = n2 / m(2);
    if ((mod(n1, 1) ~= 0) || (mod(n2, 1) ~= 0))
        error('Supplied block size is invalid; matrix cannot have m(1) x m(2) blocks');
    end
    
    % perform block transpose
    B = zeros(n1*m(2), n2*m(1));
    for i = 1:n1
        
        % get current rows
        row_idx = (i-1)*m(1) + (1:m(1));
        for j = 1:n2
            
            % get current columns
            col_idx = (j-1)*m(2) + (1:m(2));
            
            % new indices
            new_row_idx = (i-1)*m(2) + (1:m(2));
            new_col_idx = (j-1)*m(1) + (1:m(1));
            
            B(new_row_idx, new_col_idx) = A(row_idx, col_idx).';
        end
    end

end