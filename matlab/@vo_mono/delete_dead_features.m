<<<<<<< HEAD:@vo_mono/delete_dead_features.m
function obj = delete_dead_features( obj )

i = 1;
while i <= obj.nFeature
	if obj.features(i).life > 0
		i = i + 1;
	else
		obj.features = [obj.features(1:i-1) obj.features(i+1:end)];
		obj.nFeature = obj.nFeature - 1;
	end
=======
function obj = delete_dead_features( obj )

try
	idx = find([obj.features(:).life] > 0);
	nIdx = length(idx);
	obj.features = obj.features(idx);
	obj.nFeature = nIdx;
catch
	
>>>>>>> debug:matlab/@vo_mono/delete_dead_features.m
end