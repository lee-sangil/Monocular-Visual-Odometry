function obj = virtual_run( obj, pkg )

obj.step = obj.step + 1;

% Get current image
[features, points, id] = get_current_feature(pkg);
obj.cur_image = zeros(fliplr(obj.params.imSize));

obj.refresh();

if obj.virtual_set_features(features, points, id) && ... % Extract and update features
		obj.calculate_essential() && ... % RANSAC for calculating essential/fundamental matrix
		obj.calculate_motion() % Extract rotational and translational from fundamental matrix
		
	% Update vo object
	obj.backup();
	
else
	obj.reload();
end