function obj = run( obj, pkg )

% Update vo object
obj.step = obj.step + 1;

% Get current image
[~, image] = get_current_image(pkg);
obj.set_image(image);

% Extract and update features 
if obj.initialized
	obj.process();
else
	obj.initialize();
end