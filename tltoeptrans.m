function L = tltoeptrans(a)
% TLTOEPTRANS  Transformation of a two-level Toeplitz matrix into a
%              corresponding two-level Loewner form
%
%   T = TLTOEPTRANS(A) will return an MN x MN two-level Loewner matrix
%   having an N x N pattern of M x M blocks and whose coefficients are
%   determined by Fourier-like transformations of the (2M-1) x (2N-1)
%   coefficient matrix A.
%
%       See also tltoep
%
%       Requires <a href="matlab:help transbw">transbw</a>
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  1.0.0
%   Date:     22-Apr-2013
%
%

    if (isempty(a))
        error('MATLAB:tltoeptrans:emptyInput', ...
            'Input coefficient matrix must be non-empty.');
    end
    if (nnz(mod(size(a), 2)) ~= 2)
        error('MATLAB:tltoeptrans:badInput', ...
            'Input coefficient matrix must have odd dimensions.');
    end
    if (nnz(size(a) == 1))
        L = toeptrans(a(:));
        return;
    end

    % determine level sizes
    m = (size(a, 1) + 1) / 2;
    n = (size(a, 2) + 1) / 2;
    
    % create aliased version for transforms
    awrap = [zeros(1, 2*n); zeros(2*m-1, 1), a];
    SLeft = [speye(m), speye(m)];
    SRight = [speye(n); speye(n)];
    
    % diagonal matrices for computing transforms
    D1 = spdiags(exp(1j*pi/m*((-m):(m-1))'), 0, 2*m, 2*m);
    D2 = spdiags(exp(1j*pi/n*((-n):(n-1))'), 0, 2*n, 2*n);
    
    % store transformed coefficients
    S = cell(4, 1);
    S{1} = ifft2(SLeft*D1*awrap*D2*SRight);
    S{2} = ifft2(SLeft*awrap*D2*SRight);
    S{3} = ifft2(SLeft*D1*awrap*SRight);
    S{4} = ifft2(SLeft*awrap*SRight);
    
    % denominator parameters
    omplus = exp(1j * 2 * pi / n * (0:(n - 1))');
    omminus = omplus * exp(1j*pi/n); 
    phiplus = exp(1j * 2 * pi / m * (0:(m - 1))');
    phiminus = phiplus * exp(1j*pi/m);
    
    % build denominator
    on = ones(n, 1);
    C1 = (omminus * on' - on * omplus.');
    om = ones(m, 1);
    C2 = (phiminus * om' - om * phiplus.');
    Denom = kron(C1, C2);
    
    % build numerator
    ob = ones(m*n, 1);
    Num = transbw([S{2}(:), ob]*[ob'; S{3}(:).'], m);
    Num = Num + [S{1}(:), ob]*[ob'; S{4}(:).'];
    
    L = Num ./ Denom;
    L = L * spdiags(kron(omplus, phiplus), 0, m*n, m*n);
    
end