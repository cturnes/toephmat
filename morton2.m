function k = morton2(m, n)
% MORTON2  Morton re-ordering of a 2-D matrix
%
%   MORTON2(M,N) will return a index array to be used in a morton re-ordering
%   of the columns and rows of a two-level matrix whose block size is M x M
%   and whose block pattern size is N x N.  With the returned index array,
%   one may construct a permutation matrix
%
%       P = SPARSE(MORTON2(M,N), (1:(M*N)), 1, M*N, M*N);
%
%   such that P'*A*P will produce a Morton re-ordering of the columns and rows.
%
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  1.0.0
%   Date:     21-Apr-2014
%
%

    if (nargin < 2)
        n = m;
    end
    k = (1:(m*n));
    k = subdiv(reshape(k, [m n]));
    k = k(:);

end

function R = subdiv(R)

    if (numel(R) == 1)
        
        return;
        
    else
        
        [m, n] = size(R);
        
        if (m > n)
            
           Rs = cell(2, 1);
           mh = ceil(size(R, 1) / 2);
           Rs{1} = R(1:mh, :);
           Rs{2} = R((mh+1):end, :);
           
           R = [subdiv(Rs{1}); subdiv(Rs{2})];
           
        elseif (m < n)
            
            Rs = cell(2, 1);
            nh = ceil(size(R, 2) / 2);
            Rs{1} = R(:, 1:nh);
            Rs{2} = R(:, (nh+1):end);
            
            R = [subdiv(Rs{1}); subdiv(Rs{2})];
            
        else
           
           Rs = cell(4, 1);
           mh = ceil(size(R, 1) / 2);
           nh = ceil(size(R, 2) / 2);

           Rs{1} = R(1:mh, 1:nh);
           Rs{2} = R(1:mh, (nh+1):end);
           Rs{3} = R((mh+1):end, 1:nh);
           Rs{4} = R((mh+1):end, (nh+1):end);

           R = [subdiv(Rs{1}); ...
                subdiv(Rs{2}); ...
                subdiv(Rs{3}); ...
                subdiv(Rs{4}) ];
            
        end

    end

end
