% HMINV  Compute the inverse of a hierarchical matrix (MEX)
%
%   G = HMINV(H) will return a hierarchical structure corresponding to the
%   inverse matrix of the hierarchical input H.  H must have two fields --
%   'data' and 'meta' -- that prescribe the construction of the hierarchical
%   structure.  The dimensions of the input X and the matrix H must agree.
%
%
%       See also dense2hm hmtimes
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  1.0.0
%   Date:     24-Apr-2014
%
%