function obj = text_write(obj, imStep)

fid = fopen(sprintf('log/txt_features%02d.txt', imStep), 'w');

i = 1;
j = 1;
while j <= obj.features(end).id
	if obj.features(i).id ~= j
		fprintf(fid, '\r\n');
		j = j + 1;
	else
		fprintf(fid, '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t\r\n', obj.features(i).id, obj.features(i).uv1(1), obj.features(i).uv1(2), obj.features(i).uv2(1), obj.features(i).uv2(2), obj.features(i).point(1), obj.features(i).point(2), obj.features(i).point(3));
		i = i + 1;
		j = j + 1;
	end
end

fclose(fid);