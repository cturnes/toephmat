cpp_files = { [ pwd, '/dense2hm.cpp' ]; ...
              [ pwd, '/hmtimes.cpp' ]; ...
              [ pwd, '/hminv.cpp' ]; ...
            };

if (~exist('start', 'var'))
    start = 1;
end
if (~exist('stop', 'var'))
    stop = length(cpp_files);
end
supp_files = { [pwd, '/kernel/misc.cpp']; ...
               [pwd, '/kernel/misc.cpp']; ...
               [pwd, '/kernel/misc.cpp']; ...
             };

incl_dir = [pwd, '/kernel'];

mex_cmd = cell(length(cpp_files), 1);
for k = 1:length(cpp_files)
    mex_cmd{k} = ['mex ', cpp_files{k}, ' ', supp_files{k}, ' -I', incl_dir];
end

for k = start:stop
    eval(mex_cmd{k});
end