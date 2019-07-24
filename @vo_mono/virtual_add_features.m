function obj = virtual_add_features( obj, features, points, id )

max_features = obj.bucket.max_features;
obj.update_bucket();

failure = 0;
while obj.nFeature < max_features
	flag = obj.virtual_add_feature( features, points, id );
	if flag == false
		failure = failure + 1;
	else
		failure = 0;
	end
	if failure > 10
		break;
	end
end