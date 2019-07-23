function flag = virtual_set_features( obj, features, points, id )

% Update features using KLT tracker
if obj.virtual_update_features(features, points, id)
	
	% Delete features which is failed to track by KLT tracker
	obj.delete_dead_features();
	
	% Add features to the number of the lost features
	obj.virtual_add_features(features, points, id);
	
	flag = true;
	
else
	
	flag = false;

end