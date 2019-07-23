function obj = virtual_add_features( obj, features, points, id )

max_features = obj.bucket.max_features;
obj.update_bucket();

while obj.nFeature < max_features
	obj.virtual_add_feature( features, points, id );
end