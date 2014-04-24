% HMTIMES  Multiply an input by a hierarchical matrix (MEX)
%
%   Y = HMTIMES(H,X) will multiply the input matrix or vector X by the 
%   hierarchical matrix parameterized by the structure H.  H must have two
%   fields -- 'data' and 'meta' -- that prescribe the construction of the
%   hierarchical structure.  The dimensions of the input X and the matrix
%   H must agree.
%
%
%       See also dense2hm hminv
%
%   Christopher K. Turnes
%   Georgia Institute of Technology
%   Version:  1.0.0
%   Date:     24-Apr-2014
%
%