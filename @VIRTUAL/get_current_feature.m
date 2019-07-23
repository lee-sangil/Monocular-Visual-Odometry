function [feature, point, id] = get_current_feature(obj)

if obj.step < obj.imLength
	obj.step = obj.step + 1;
	
	feature = obj.features{obj.step};
	point = obj.points{obj.step};
	id = obj.points_id{obj.step};
	
else
	error('reach the end of file');
end

if obj.step == obj.imLength
	obj.eof = true;
end

end