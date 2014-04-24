function L = toeptrans2hmat(a, nlim)
% TOEPTRANS2HMAT  Transformation of a Toeplitz matrix into a corresponding 
%                 Loewner form stored as a hierarchical structure
%
%   T = TOEPTRANS2HMAT(A) will return a hierarchical structure representing 
%   the N x N Loewner matrix whose coefficients are determined by Fourier-
%   like transformations of the (2N-1) x 1 coefficient matrix A.
%
%   T = TOEPTRANS2HMAT(A,NLIM) will perform the same, but will stop
%   subdivision at NLIM x NLIM blocks.
%
%       See also toep toeptrans
%
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  1.0.0
%   Date:     22-Apr-2013
%
%

    if (isempty(a))
        error('MATLAB:toeptrans2hmat:emptyInput', ...
            'Input coefficient array must be non-empty.');
    end
    if (mod(numel(a), 2) == 0)
        error('MATLAB:toeptrans2hmat:badInput', ...
            'Input coefficient array must have odd length.');
    end
    if (nargin < 2)
        nlim = 32;
    end
    
    % get size
    n = (size(a, 1) + 1) / 2;
    
    % create generator arrays
    u = zeros(n, 2);
    u(1, 2) = -1;
    u(:, 1) = a(n:end) - [0; a(1:(n-1))];
    u = ifft(spdiags(exp(1j*pi/n*(0:(n-1))'), 0, n, n) * u) * sqrt(n);
    v = zeros(n, 2);
    v(n, 1) = -1;
    v(:, 2) = flipud(conj(a(n:end) + [0; a(1:(n-1))]));
    v = ifft(v) * sqrt(n);
    
    % denominator terms
    omplus = exp(1j * 2 * pi / n * (0:(n-1))');
    omminus = omplus * exp(1j*pi/n);
    
    % create structure
    L.data = zeros(min(n^2, 4096^2), 1);
    L.meta = zeros(11, 0);
    
    % get data
    [L, dataOffset] = subdiv(L, u, v, omplus, omminus, 1, 1, n, nlim, 1);
    L.data = L.data(1:(dataOffset-1));
    L.meta = int32(L.meta);

end

function [L, dataOffset] = subdiv(L, u, v, omplus, omminus, rI, cI, n, nlim, dataOffset)

    % this meta index
    midx = size(L.meta, 2) + 1;
    
    % matrix is small enough to go dense
    if (n <= nlim)
        
        % build matrix
        idxr = rI:(rI+n-1);
        idxc = cI:(cI+n-1);
        Num = u(idxr, :)*v(idxc, :)';
        on = ones(n, 1);
        Denom = omminus(idxr)*on' - on*omplus(idxc).';
        M = Num ./ Denom;
        
        % update meta data
        L.meta(:, midx) = [0, n, n, 0, 0, dataOffset-1, ...
                               -1, -1, -1, n, -1]';
        M = M(:);
        Lm = numel(M);
        dataIdx = (dataOffset):(dataOffset + Lm - 1);
        L.data(dataIdx) = M(:);
        dataOffset = dataOffset + Lm;
        
    else
        
        L.meta(:, midx) = zeros(11, 1);
        
        
        % set some initial values of the meta data array
        L.meta(1, midx) = 1;
        
        % determine subdivision
        nh1 = ceil(n / 2);
        nh2 = n - nh1;
        idxr1 = rI:(rI+nh1-1);
        idxr2 = (rI+nh1):(rI+n-1);
        idxc1 = cI:(cI+nh1-1);
        idxc2 = (cI+nh1):(cI+n-1);
        on1 = ones(nh1, 1);
        on2 = ones(nh2, 1);
        L.meta(2:5, midx) = [nh1, nh1, nh2, nh2];
        
        % low-rank data from northeast corner
        Csub = (u(idxr1, :)*v(idxc2, :)') ./ (omminus(idxr1)*on2' - on1*omplus(idxc2).');
        if (nh2 >= 64)
            % use denominator as base
            [U, S, V] = lrsvd(Csub, 0.5);
        else
            % compute svd on full matrix term
            [U, S, V] = svd(Csub, 0);
        end
        % determine rank:
        neRank = nnz(diag(S, 0) >= (S(1)*eps*nh1));
        L.meta(10, midx) = neRank;
        S = sqrt(S(1:neRank, 1:neRank));
        U = U(:, 1:neRank) * S;
        V = V(:, 1:neRank) * S;
        
        L.meta(6, midx) = dataOffset - 1;
        LUV = numel(U) + numel(V);
        dataIdx = (dataOffset):(dataOffset + LUV - 1);
        L.data(dataIdx) = [U(:); V(:)];
        dataOffset = dataOffset + LUV;
        clear U S V Csub dataIdx;
        
        % low-rank data from southwest corner
        Csub = (u(idxr2, :)*v(idxc1, :)') ./ (omminus(idxr2)*on1' - on2*omplus(idxc1).');
        if (nh2 >= 64)
            % use denominator as base
            [U, S, V] = lrsvd(Csub, 0.5, neRank + 1);
        else
            % compute svd on full matrix term
            [U, S, V] = svd(Csub, 0);
        end
        % determine rank:
        swRank = nnz(diag(S, 0) >= (S(1)*eps*nh1));
        L.meta(11, midx) = swRank;
        S = sqrt(S(1:swRank, 1:swRank));
        U = U(:, 1:swRank) * S;
        V = V(:, 1:swRank) * S;
        
        L.meta(7, midx) = dataOffset - 1;
        LUV = numel(U) + numel(V);
        dataIdx = (dataOffset):(dataOffset + LUV - 1);
        L.data(dataIdx) = [U(:); V(:)];
        dataOffset = dataOffset + LUV;
        clear U S V Csub dataIdx idxr1 idxr2 idxc1 idxc2 on1 on2;
        
        % set the meta index for the next chunk
        L.meta(8, midx) = midx;
        [L, dataOffset] = subdiv(L, u, v, omplus, omminus, rI, cI, nh1, nlim, dataOffset);
        
        % set the meta index for the next chunk
        L.meta(9, midx) = size(L.meta, 2);
        [L, dataOffset] = subdiv(L, u, v, omplus, omminus, rI + nh1, cI + nh1, nh2, nlim, dataOffset);
        
    end
                                         
end