function obj = add_features( obj )

max_features = obj.bucket.max_features;
obj.update_bucket();

while obj.nFeature < max_features
	obj.add_feature();
end