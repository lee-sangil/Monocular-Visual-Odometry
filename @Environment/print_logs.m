function obj = print_logs( obj, timePassed )

file_logger = fopen(sprintf([obj.params.filepath 'log_%04d.txt'], obj.vo.step), 'w');

fprintf(file_logger, '# id\tlife\tuv\txyz\r\n');
for i = 1:length(obj.vo.features)
	if obj.vo.features(i).life ~= 0
		fprintf(file_logger, '%d\t%d\t%d\t%4.4f\t%4.4f\t%.5f\t%.5f\t%.5f\r\n', ...
			obj.vo.features(i).id, obj.vo.features(i).life, obj.vo.features(i).is_wide, ...
			obj.vo.features(i).uv(1), obj.vo.features(i).uv(2), ...
			obj.vo.features(i).point(1), obj.vo.features(i).point(2), ...
			obj.vo.features(i).point(3));
	end
end

fclose(file_logger);