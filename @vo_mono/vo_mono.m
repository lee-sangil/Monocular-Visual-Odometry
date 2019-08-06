classdef vo_mono < handle
	properties (GetAccess = public, SetAccess = private)
		
		identifier
		
		undist_image
		curr_image
        prev_image
		
		initialized = false
		processed = false
		
		state
		pose
		
		% Iteration
		step = 0
		
		% Constant
		params
		
% 		% Debug
% 		groundtruth
	end
	methods (Access = public)
		% CONSTRUCTOR
		function obj = vo_mono(pkg)
			
			obj.identifier = 'mono';
			
% 			if isprop(pkg, 'pose')
% 				obj.groundtruth = pkg.pose;
% 			end
			
            % Save features attibutes
            obj.params.saveDir = 'logs';
			if ~exist(['./' obj.params.saveDir], 'dir')
				if ispc
					command = ['mkdir ' obj.params.saveDir];
				elseif isunix
					command = ['mkdir -p ' obj.params.saveDir];
				end
				system(command);
			end
            
			%% [Image]
			obj.params.imSize = pkg.imSize;
			obj.params.intrinsic = pkg.K;
			obj.params.radialDistortion = pkg.radialDistortion;
			obj.params.tangentialDistortion = pkg.tangentialDistortion;
			
			%% [Bootstrap]
			% Bootsrap frames
			obj.params.bootstrap.frames = [1, 3];
			
			% For harris keypoint selector
			obj.params.bootstrap.harris.patch_size = 9;
			obj.params.bootstrap.harris.kappa = 0.08;
			obj.params.bootstrap.harris.num_keypoints = 500;
			obj.params.bootstrap.harris.nonmaximum_supression_radius = 20;
			
			% For KLT tracking
			obj.params.bootstrap.KLT.num_pyramid_levels = 3;
			obj.params.bootstrap.KLT.max_bidirectional_error = 1;
			obj.params.bootstrap.KLT.block_size = [19 19];
			obj.params.bootstrap.KLT.max_iterations = 60;
			
			% For RANSAC
			obj.params.bootstrap.RANSAC.num_iterations = 50;
			obj.params.bootstrap.RANSAC.pixel_tolerance = 3;
			
			%% [ProcessFrame]
			% For KLT tracker
			% See: https://ch.mathworks.com/help/vision/ref/vision.pointtracker-system-object.html
			obj.params.process.KLT.num_pyramid_levels = 3;
			obj.params.process.KLT.max_bidirectional_error = 6;
			obj.params.process.KLT.block_size = [31 31];
			obj.params.process.KLT.max_iterations = 30;
			
			% For P3P algorithm
			obj.params.process.p3p.num_iterations = 200;
			obj.params.process.p3p.pixel_tolerance = 20;
			obj.params.process.p3p.min_inlier_count = 6;
			
			% For adding new landmarks
			obj.params.process.landmarks.bearing_min = 0.4 * pi/180;
			obj.params.process.landmarks.bearing_max = 4.0 * pi/180;
			obj.params.process.new_candidate_tolerance = 8;
			
			% For harris keypoint selection and matching
			obj.params.process.harris.patch_size = 9;
			obj.params.process.harris.kappa = 0.08;
			obj.params.process.harris.num_keypoints = 300;
			obj.params.process.harris.nonmaximum_supression_radius = 20;
			
			% For feature matching
			obj.params.process.harris.descriptor_radius = 9;
			obj.params.process.harris.match_lambda = 10;
			
			%% [Additional Parameters]
			obj.params.additional_fix.flag = true;
			obj.params.additional_fix.bootstrap_interval = 20;
			obj.params.debug = 1;
			obj.params.initScale = 1;
		end
        
		% Set image
        obj = set_image( obj, image )
		
		% Script operations
		obj = initialize( obj )
		obj = process( obj )
		obj = run( obj, pkg )
        
% 		% Test for Virtual dataset
% 		obj = virtual_run( obj, pkg )
% 		flag = virtual_set_features( obj, features, points, id )
% 		flag = virtual_update_features( obj, features, points, id )
% 		obj = virtual_delete_dead_features( obj, features, points )
% 		obj = virtual_add_features( obj, features, points, id )
% 		flag = virtual_add_feature( obj, features, points, id )
	end
end