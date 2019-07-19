function obj = delete_dead_features( obj )

i = 1;
while i <= obj.nFeature
	if obj.features(i).life > 0
		i = i + 1;
	else
		obj.features = [obj.features(1:i-1) obj.features(i+1:end)];
		obj.nFeature = obj.nFeature - 1;
	end
end