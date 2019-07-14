baseDir = 'F:\Datasets\RGBD\odometry_5\';
rgb = importData([baseDir 'rgb.txt'], '%f%*s%[^\n\r]', 4);
depth = importData([baseDir 'depth.txt'], '%f%*s%[^\n\r]', 4);

cd([baseDir 'rgb']);

for i = 1:length(rgb)
	movefile(sprintf('%d.png', rgb(i)), sprintf('%10.6f.png', rgb(i)/1000000.0))
end

cd([baseDir 'depth']);

for i = 1:length(depth)
	movefile(sprintf('%d.png', depth(i)), sprintf('%10.6f.png', rgb(i)/1000000.0))
end

fid = fopen([baseDir 'rgb_.txt'], 'w');
fprintf(fid, '# color images\n# file\n# timestamp filename\n');
for i = 1:length(rgb)
	fprintf(fid, '%10.6f rgb/%10.6f.png\n', rgb(i)/1000000.0, rgb(i)/1000000.0);
end
fclose(fid);

fid = fopen([baseDir 'depth_.txt'], 'w');
fprintf(fid, '# depth images\n# file\n# timestamp filename\n');
for i = 1:length(depth)
	fprintf(fid, '%10.6f depth/%10.6f.png\n', rgb(i)/1000000.0, rgb(i)/1000000.0);
end
fclose(fid);

fid = fopen([baseDir 'associations.txt'], 'w');
for i = 1:length(rgb)
	fprintf(fid, '%10.6f rgb/%10.6f.png %10.6f depth/%10.6f.png\n', rgb(i)/1000000.0, rgb(i)/1000000.0, rgb(i)/1000000.0, rgb(i)/1000000.0);
end
fclose(fid);