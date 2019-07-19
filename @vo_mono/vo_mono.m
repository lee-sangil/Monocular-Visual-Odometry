classdef vo_mono < handle
	properties (Access = private)
		
		identifier
		
		cur_image
		prev_image
		
		bucket
		features
		tracker
		tracker_initialized
		
		features_backup
		tracker_backup
		
		nFeature
		nFeatureMatched
		nFeature2DInliered
		nFeature3DReconstructed
		nFeatureInlier
		new_feature_id
		
		abs_scale
		scale_initialized
		
		essential
		TRec
		TocRec
		PocRec

		% iteration
		step
		stage
		
		% Constant
		params
	end
	methods (Access = public)
		% CONSTRUCTOR
		function obj = vo_mono(pkg)

			obj.identifier = 'mono';
			
			% image
			obj.params.imSize = get_im_size(pkg);
			obj.params.K = get_intrinsic(pkg);
			obj.params.thInlier = 10;
			obj.scale_initialized = false;
			
			% bucket
			obj.bucket.grid = [ 32, 8 ]; % [width, height]
			obj.bucket.mass = zeros(obj.bucket.grid);
			obj.bucket.size = [floor(obj.params.imSize(1)/obj.bucket.grid(1)), floor(obj.params.imSize(2)/obj.bucket.grid(2))];
			obj.bucket.max_features = 100; % floor(sqrt(prod(imSize)));
			
			% Create a tracker object.
			obj.tracker = vision.PointTracker('MaxBidirectionalError', 1);
			obj.tracker_initialized = false;
			
			% Descriptor
			obj.params.descriptor = @feature_harris;

			% variables
			obj.nFeature = 0;
			obj.nFeatureMatched = 0;
			obj.nFeature2DInliered = 0;
			obj.nFeature3DReconstructed = 0;
			obj.nFeatureInlier = 0;
			obj.new_feature_id = 1;
			obj.step = 1;

			% initial position
			obj.TRec{1} = eye(4);
			obj.TocRec{1} = eye(4);
			obj.PocRec(:,1) = zeros(4,1);
			
			% make Gaussian kernel
			std = 3;
			sz = 3*std;
			d = (repmat(-sz : sz, 2*sz+1,1).^2 + (repmat(-sz : sz, 2*sz+1,1).^2)').^0.5;
			obj.params.kernel = normpdf(d, zeros(size(d)), std*ones(size(d)));
		end

		% Add Listener
% 		function lh = createListener(obj)
% 			lh = addlistener(obj, 'RELOAD', @reload);
% 		end

		% GET functions
		nFeature = get_nFeature( obj )
		nFeatureMatched = get_nFeatureMatched( obj )
		nFeature2DInliered = get_nFeature2DInliered( obj )
		nFeature3DReconstructed = get_nFeature3DReconstructed( obj )
		nFeatureInlier = get_nFeatureInlier( obj )

		% Script operations
		obj = backup( obj )
		obj = reload( obj )
		obj = run( obj, pkg )
		obj = update( obj )
		obj = plot_state( obj, plot_initialized )
		obj = text_write( obj, imStep )
		
		% Set image
        obj = set_image( obj, image )
		
		% Feature operations
		flag = extract_features( obj )
		flag = update_features( obj )
		obj = delete_dead_features( obj )
		obj = add_features( obj )
		obj = add_feature( obj )
		
		% Calculations
		flag = calculate_essential( obj )
		flag = calculate_motion( obj )
		[R, t] = verify_solutions( obj, R_vec, t_vec )
		[R, t] = scale_propagation( obj, R, t )

		% ETC
		obj = update_bucket( obj )
		[idx, length] = seek_index( obj, max_length, condition )
	end
end