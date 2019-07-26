function obj = run( obj, pkg )

% Get current image
[~, image] = get_current_image(pkg);
obj.set_image(image);
	
obj.refresh();

% Extract and update features 
if obj.extract_features()
	
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
else
	obj.reload();
	
end