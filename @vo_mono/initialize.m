function obj = initialize( obj )

if isempty(obj.prev_image)
	return;
end

fprintf('\nWaiting for Bootstrapping ...\n');

[landmarks, I2_keypts] = bootstrap(obj.prev_image, obj.curr_image, obj.params.bootstrap, obj.params.intrinsic);

% Create initial state using the output of bootstrapping
obj.state.P = I2_keypts;   % NOTE: M x 2 with [u, v] notation
obj.state.X = landmarks;   % NOTE: M x 3
obj.state.C = [];
obj.state.F = [];
obj.state.T = [];
obj.prev_image = obj.curr_image;

fprintf('Bootstrap finished !\n');
obj.initialized = true;