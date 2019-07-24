classdef vo_mono < handle
	properties (Access = private)
		
		identifier
		
		undist_image
		cur_image
        prev_image
        
		bucket
		features
		features_backup
		
		scale_initialized
		
		nFeature
		nFeatureMatched
		nFeature2DInliered
		nFeature3DReconstructed
		nFeatureInlier
		new_feature_id
		
		R_vec
		t_vec
		TRec
		TocRec
		PocRec
		
		% Iteration
		step = 1
		
		% Constant
		params
	end
	methods (Access = public)
		% CONSTRUCTOR
		function obj = vo_mono(pkg)
			
			obj.identifier = 'mono';
			
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
            
			% Image
			obj.params.imSize = pkg.imSize;
			obj.params.K = pkg.K;
			obj.params.radialDistortion = pkg.radialDistortion;
			obj.params.tangentialDistortion = pkg.tangentialDistortion;
			obj.scale_initialized = false;
			
			% Bucket
			obj.bucket.safety = 20;
			obj.bucket.grid = [ 32, 8 ]; % [width, height]
			obj.bucket.mass = zeros(obj.bucket.grid);
			obj.bucket.prob = zeros(obj.bucket.grid);
			obj.bucket.size = [floor(obj.params.imSize(1)/obj.bucket.grid(1)), floor(obj.params.imSize(2)/obj.bucket.grid(2))];
			obj.bucket.max_features = 500;
			
			% Variables
			obj.nFeature = 0;
			obj.nFeatureMatched = 0;
			obj.nFeature2DInliered = 0;
			obj.nFeature3DReconstructed = 0;
			obj.nFeatureInlier = 0;
			obj.params.thInlier = 5;
			obj.new_feature_id = 1;
			
			% Initial position
			obj.TRec{1} = eye(4);
			obj.TocRec{1} = eye(4);
			obj.PocRec(:,1) = zeros(4,1);
			
			% Gaussian kernel
			std = 3;
			sz = 3*std;
			d = (repmat(-sz : sz, 2*sz+1,1).^2 + (repmat(-sz : sz, 2*sz+1,1).^2)').^0.5;
			obj.params.kernel = normpdf(d, zeros(size(d)), std*ones(size(d)));
			
			% RANSAC parameter
			ransacCoef_calc_essential.iterMax = 1e4;
			ransacCoef_calc_essential.minPtNum = 5;
			ransacCoef_calc_essential.thInlrRatio = 0.9;
			ransacCoef_calc_essential.thDist = 1e-7;
			ransacCoef_calc_essential.thDistOut = 1e-7;
			ransacCoef_calc_essential.funcFindF = @fivePoint;
			ransacCoef_calc_essential.funcDist = @obj.essential_model_error;
			
			ransacCoef_scale_prop.iterMax = 1e3;
			ransacCoef_scale_prop.minPtNum = obj.params.thInlier;
			ransacCoef_scale_prop.thInlrRatio = 0.9;
			ransacCoef_scale_prop.thDist = 0.5; % standard deviation
			ransacCoef_scale_prop.thDistOut = inf; % three times of standard deviation
			ransacCoef_scale_prop.funcFindF = @obj.calculate_scale;
			ransacCoef_scale_prop.funcDist = @obj.calculate_scale_error;
			
			obj.params.ransacCoef_calc_essential = ransacCoef_calc_essential;
			obj.params.ransacCoef_scale_prop = ransacCoef_scale_prop;
			
			% Statistical model
			obj.params.var_theta = (90 / 1241 / 2)^2;
			obj.params.var_point = 1;
		end
		
		% Add Listener
% 		function lh = createListener(obj)
% 			lh = addlistener(obj, 'RELOAD', @reload);
% 		end
		
		% GET functions
		identifier = get_identifier( obj )
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
		obj = print_logs( obj, pkg, timePassed )
		
        % Set image
        obj = set_image( obj, image )
        
		% Feature operations
		[points, validity] = klt_tracker( obj )
		flag = extract_features( obj )
		flag = update_features( obj )
		obj = delete_dead_features( obj )
		obj = add_features( obj )
		obj = add_feature( obj )
		
		% Calculations
		flag = calculate_essential( obj )
		flag = calculate_motion( obj )
		[R, t] = verify_solutions( obj, R_vec, t_vec )
		[R, t, flag, inlier, outlier] = scale_propagation( obj, R, t )
		[R, t, flag, inlier] = findPoseFrom3DPoints( obj )
		[X_prev, X_curr, lambda_prev, lambda_curr] = contructDepth( obj, x_prev, x_curr, R, t )
		obj = update3DPoints( obj, R, t, inlier, outlier )
		
		% RANSAC
		E = eight_point_essential_hypothesis( obj, x1, x2 )
		dist = essential_model_error( obj, E, x1, x2 )
		scale = calculate_scale( obj, P1, P1_ref )
		dist = calculate_scale_error( obj, scale, P1, P1_ref )
		
		% ETC
		obj = update_bucket( obj )
		[idx, length] = seek_index( obj, max_length, condition )
		
		% Test for Virtual dataset
		obj = virtual_run( obj, pkg )
		flag = virtual_set_features( obj, features, points, id )
		flag = virtual_update_features( obj, features, points, id )
		obj = virtual_delete_dead_features( obj, features, points )
		obj = virtual_add_features( obj, features, points, id )
		flag = virtual_add_feature( obj, features, points, id )
	end
end