function obj = delete_dead_features( obj )

try
	idx = find([obj.features(:).life] > 0);
	nIdx = length(idx);
	obj.features = obj.features(idx);
	obj.nFeature = nIdx;
catch
	
end