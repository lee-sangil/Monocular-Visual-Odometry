function obj = virtual_run( obj, pkg )

% Get current image
[features, points, id] = get_current_feature(pkg);
obj.cur_image = zeros(fliplr(obj.params.imSize));

obj.refresh();

obj.virtual_set_features(features, points, id);

% RANSAC for calculating essential/fundamental matrix
if obj.calculate_essential()
	
	% Extract rotational and translational from fundamental matrix
	if obj.calculate_motion()
		
		% Update vo object
		obj.backup();
		obj.step = obj.step + 1;
		
	else
		% Load features and tracker backed up
		obj.reload();
		
	end
else
	obj.reload();
	
end