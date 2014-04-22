%{
    @(#)File:          /compile_kernel.m
    @(#)Package:       Hierarchical Matrix Inversion
    @(#)Version:       1.0.0
    @(#)Last changed:  21 April 2014
    @(#)Author:        C. Turnes
    @(#)Copyright:     Georgia Institute of Technology
    @(#)Purpose:       Compiles the MEX functions for the Hierarchical
                       matrix inversion package
%}

% source files
cpp_files = { [ pwd, '/dense2hm.cpp' ]; ...
              [ pwd, '/hmtimes.cpp' ]; ...
              [ pwd, '/hminv.cpp' ]; ...
            };

% check if we are only compiling a subset
if (~exist('start', 'var'))
    start = 1;
end
if (~exist('stop', 'var'))
    stop = length(cpp_files);
end

% supplemental files that must be compiled
supp_files = { [pwd, '/kernel/misc.cpp']; ...
               [pwd, '/kernel/misc.cpp']; ...
               [pwd, '/kernel/misc.cpp']; ...
             };

% include directory
incl_dir = [pwd, '/kernel'];

% construct mex compile commands
mex_cmd = cell(length(cpp_files), 1);
for k = 1:length(cpp_files)
    mex_cmd{k} = ['mex ', cpp_files{k}, ' ', supp_files{k}, ' -I', incl_dir];
end

% compile the functions
for k = start:stop
    eval(mex_cmd{k});
end