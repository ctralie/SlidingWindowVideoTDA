addpath(genpath('toolbox-master'));
actions = {'high_jump', 'diving_springboard_3m', 'discus_throw'};

for ii = 1:length(actions)
    counter = 0;
    files = dir(actions{ii});
    for kk = 1:length(files)
        [~, b, fext] = fileparts(files(kk).name);
        if strcmp(fext, '.seq')
            fprintf(1, 'Processing %s...\n', files(kk).name);
            dirName = sprintf('%s/%i', actions{ii}, counter);
            if exist(dirName) == 0
                mkdir(dirName);
                sq = seqIo(sprintf('%s/%s', actions{ii}, files(kk).name), 'reader');
                N = length(sq.getts());
                for jj = 1:N
                    I = sq.getnext();
                    imwrite(I, sprintf('%s/%i.png', dirName, jj-1));
                end
                command = sprintf('avconv -r 30 -i %s/%s.png -r 30 -b 30000k %s/video.ogg', dirName, '%d', dirName)
                system(command);
            end
            counter = counter + 1;
        end
    end
end