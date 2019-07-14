clc
baseDir = 'F:\Datasets\RGBD\odometry_13\';
FILE = fopen([baseDir 'rgb.txt']);
fid = fopen([baseDir 'associations.txt'], 'w');
while 1
	S = fgetl(FILE);
	if ~ischar(S), break, end
	
	if S(1) ~= '#'
		frame = sscanf(S(1:17), '%f');
		fprintf(fid, '%10.6f rgb/%10.6f.png %10.6f depth/%10.6f.png\n', frame, frame, frame, frame);
	end
end

fclose(FILE);
fclose(fid);