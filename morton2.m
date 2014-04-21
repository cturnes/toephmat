function k = morton2(m, n)

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
