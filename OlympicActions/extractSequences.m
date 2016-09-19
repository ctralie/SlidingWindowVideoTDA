addpath(genpath('toolbox-master'));
actions = {'discus_throw', 'diving_springboard_3m', 'high_jum'};

for ii = 1:length(actions)
    counter = 0;
    files = dir(actions{ii});
    for kk = 1:length(files)
        [~, b, fext] = fileparts(files(kk).name);
        if strcmp(fext, '.seq')
            fprintf(1, 'Processing %s...\n', files(kk).name);
            dirName = sprintf('%s/%i', actions{ii}, counter);
            mkdir(dirName);
            sq = seqIo(sprintf('%s/%s', actions{ii}, files(kk).name), 'reader');
            N = length(sq.getts())
            for jj = 1:N
                I = sq.getnext();
                imwrite(I, sprintf('%s/%i.png', dirName, jj-1));
            end
            counter = counter + 1;
        end
    end
end