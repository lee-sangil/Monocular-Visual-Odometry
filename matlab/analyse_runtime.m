fid = fopen('/media/icsl/Samsung_Duo/log.txt','r');

processes = [];
runtimes = [];
time_ref = 0;
while( ~feof(fid) )
    s = fgetl(fid);
    if s(1) == '#' && s(2) ~= '#'
        ss = split(s, ' ');
        proc = join([ss(2:end-2); {ss{end-1}(1:end-1)}],' ');
        time = str2num(ss{end});% - time_ref;
        time_ref = str2num(ss{end});
        
        if ismember(proc{1}, processes)
            idx = find(ismember(processes, proc{1}));
            runtimes{idx} = [runtimes{idx} time/1000];
        else
            processes = [processes proc];
            runtimes = [runtimes {time/1000}];
        end
    elseif s(1) == '='
        time_ref = 0;
    end
end
fclose(fid);

processes = processes([1 3 2 6 5 7 8 4]);
runtimes = runtimes([1 3 2 6 5 7 8 4]);
processes{2} = 'Cull features';
processes = processes(1:end-1);
runtimes = runtimes(1:end-1);

figure();
lineStyle = {'-','-'};
cmap = [lines(7);0.5*lines(7)];
cmap(1,:) = 0.8*cmap(1,:);
% cmap = jet(256);
% cmap = cmap(round(linspace(1,256,length(runtimes))),:);
handle = [];
for i = 1:length(runtimes)
    [V, R] = histcounts(log10(runtimes{i}),50,'normalization','pdf');
    T = 0.5 * (10.^R(2:end) + 10.^R(1:end-1));
    if i < 8
        semilogx(T,V,'color','k','LineWidth',1,'LineStyle',lineStyle{1});hold on;
        h = area(T,V,'facecolor',cmap(i,:),'facealpha',0.5);
        handle = [handle h];
    else
        semilogx(T,V,'color','k','LineWidth',1,'LineStyle',lineStyle{2});hold on;
        h = area(T,V,'facecolor',cmap(i,:),'facealpha',0.5);
        handle = [handle h];
    end
end
legend(handle,processes);
xlabel('Execution time (ms)');
ylabel('Density');
grid on;
set(gca, 'fontsize', 13);
