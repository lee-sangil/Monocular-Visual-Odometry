baseDir = 'D:\Libraries\Documents\Google Drive\Doing\Master thesis\data\odometry_13\dso\';
FILE = fopen([baseDir 'camToWorld.txt']);

while 1
	S = fgetl(FILE);
	if ~ischar(S), break, end
	
	if length(S) > 18 && all(S(1:18) == 'OUT: Current Frame')
		frame = sscanf(S(20:end), '%d');
		
		mat = zeros(4,4);
		for i = 1:3
			S = fgetl(FILE);
			a = sscanf(S, '%f %f %f %f');
			mat(i,1:4) = a;
		end
		mat(4,4) = 1;
		abs_tform{frame} = mat;
		fprintf('frame: %d\n', frame);
	end
end

fclose(FILE);

save([baseDir 'dso.mat'], 'abs_tform');