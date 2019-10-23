function obj = add_features( obj )

max_features = obj.bucket.max_features;
obj.update_bucket();

while obj.nFeature < max_features
	obj.add_feature();
end

if obj.nFeature == max_features
	points = features2points(obj);
	if obj.tracker_initialized
		setPoints(obj.tracker, points);
	else
		initialize(obj.tracker, points, obj.cur_image);
		obj.tracker_initialized = true;
	end
end