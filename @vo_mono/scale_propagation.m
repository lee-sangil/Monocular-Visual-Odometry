function [R, t, flag, inlier, outlier] = scale_propagation( obj, R, t )

inlier = [];
outlier = [];

%% TRI-IMAGES SCALE COMPENSATION
if obj.scale_initialized
	
	% Seek index of which feature is 3D reconstructed currently,
	% and 3D initialized previously
	idx = find([obj.features(:).is_3D_reconstructed] == true & ... % for expected 3D points
				[obj.features(:).is_3D_init] == true); % for initialized 3D points
	nPoint = length(idx);	
	
	% Use RANSAC to find suitable scale
	if nPoint > obj.params.ransacCoef_scale_prop.minPtNum
		
		% Get initialized 3D point
		P1_ini = zeros(4, nPoint);
		for i = 1:nPoint
			P1_ini(:,i) = obj.TocRec{obj.step-1} \ obj.features(idx(i)).point_init;
		end
		
		% Get expected 3D point by transforming the coordinates of the
		% observed 3d point
		P1_exp = zeros(4, nPoint);
		for i = 1:nPoint
			P1_exp(:,i) = obj.features(idx(i)).point;
		end
		
		% RANSAC
% 		obj.params.ransacCoef_scale_prop.weight = [obj.features(idx).life] ./ sqrt([obj.features(idx).point_var]);
		points = [obj.features(idx).point];
		obj.params.ransacCoef_scale_prop.weight = atan(-points(3,:)/10+3)+pi/2;
		[scale, inlier, outlier] = ransac(P1_exp(1:3,:), P1_ini(1:3,:), obj.params.ransacCoef_scale_prop);
		obj.nFeatureInlier = length(inlier);
		
	end

	% Use the previous scale, if the scale cannot be found
	if nPoint <= obj.params.ransacCoef_scale_prop.minPtNum || length(inlier) < obj.params.thInlier || isempty(scale)
		warning('there are a few SCALE FACTOR INLIERS')
		idx = find([obj.features(:).is_3D_init] == true);
		nPoint = length(idx);
		
		% Use RANSAC to find suitable scale
		if nPoint > obj.params.thInlier
			objectPoints_ = [obj.features(idx).point];
			objectPoints = permute(objectPoints_(1:3,:),[3 2 1]);
			
			imagePoints_ = zeros(2,nPoint);
			for i = 1:nPoint
				imagePoints_(:,i) = obj.features(idx(i)).uv(:,1);
			end
			imagePoints = permute(imagePoints_,[3 2 1]);
			
			[r_vec, t_vec, success, inlier] = cv.solvePnPRansac(objectPoints, imagePoints, obj.params.K, 'Rvec', R, 'IterationsCount', 1e4, 'ReprojectionError', obj.params.reprojError);
			if success == false
				[r_vec, t_vec, success, inlier] = cv.solvePnPRansac(objectPoints, imagePoints, obj.params.K, 'IterationsCount', 1e4, 'ReprojectionError', obj.params.reprojError*2);
			end
			
			R = expm(skew(r_vec/2));
			t = t_vec/2;
% 			T = obj.TRec{obj.step-1} \ [R t; 0 0 0 1];
% 			R = T(1:3,1:3);
% 			t = T(1:3,4);
			
			flag = success;
			inlier = idx(transpose(inlier) + 1);
			outlier = idx(~ismember(idx,inlier));
		else
			inlier = [];
			outlier = [];
			
			scale = norm(obj.TRec{obj.step-1}(1:3,4));
			
			% Update scale
			t = scale * t;
			flag = false;
		end
		
	else
		% Update scale
		t = scale * t;
		flag = true;
	end
end

%% INITIALIZATION
% Initialze scale, in the case of the first time
if ~obj.scale_initialized
	
	scale = obj.params.initScale;
	t = scale * t;
	
	obj.nFeatureInlier = obj.nFeature3DReconstructed;
	inlier = find([obj.features(:).is_3D_reconstructed] == true);
	flag = true;
end

end