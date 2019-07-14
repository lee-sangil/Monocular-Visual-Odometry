function obj = delete_dead_features( obj )

try
	[idx, nIdx] = seek_index(obj, obj.nFeature, [obj.features(:).life] > 0);
	obj.features = obj.features(idx);
	obj.nFeature = nIdx;
catch
	
end