function [feature, point, id] = get_current_feature(obj)

if obj.step + obj.imInit <= obj.imLength
	obj.step = obj.step + 1;
	
	feature = obj.features{obj.step + obj.imInit - 1};
	point = obj.points{obj.step + obj.imInit - 1};
	id = obj.points_id{obj.step + obj.imInit - 1};
	
else 
	error('reach the end of file'); 
end 

if obj.step + obj.imInit - 1 == obj.imLength
	obj.eof = true;
end

end