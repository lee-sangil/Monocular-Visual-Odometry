function obj = run( obj, pkg )

obj.step = obj.step + 1;

% Get current image
[~, image] = get_current_image(pkg);
obj.set_image(image);
	
obj.refresh();

if obj.extract_features() && ... % Extract and update features
		obj.calculate_essential() && ... % RANSAC for calculating essential/fundamental matrix
		obj.calculate_motion() % Extract rotational and translational from fundamental matrix
		
	% Update vo object
	obj.backup();
	
else
	obj.reload();
end