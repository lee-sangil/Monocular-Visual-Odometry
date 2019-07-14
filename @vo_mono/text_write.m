function obj = text_write(obj, imStep)

fid = fopen(sprintf([obj.params.saveDir '/features%02d.txt'], imStep), 'w');

i = 1;
j = 1;
while j <= obj.features(end).id
	if obj.features(i).id ~= j
		fprintf(fid, '\r\n');
	else
		point = obj.TocRec{obj.step-1} \ obj.features(i).point;
		if size(obj.features(i).uv, 2) > 1
			fprintf(fid, '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\r\n', obj.features(i).id, obj.features(i).uv(1,2), obj.features(i).uv(2,2), obj.features(i).uv(1,1), obj.features(i).uv(2,1), point(1), point(2), point(3), obj.features(i).point_init(1), obj.features(i).point_init(2), obj.features(i).point_init(3), obj.features(i).point_var);
		else
			fprintf(fid, '%d\t%c\t%c\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\r\n', obj.features(i).id, ' ', ' ', obj.features(i).uv(1,1), obj.features(i).uv(2,1), point(1), point(2), point(3), obj.features(i).point_init(1), obj.features(i).point_init(2), obj.features(i).point_init(3), obj.features(i).point_var);
		end
		i = i + 1;
	end
	j = j + 1;
end

fclose(fid);