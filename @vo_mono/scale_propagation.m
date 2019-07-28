function [R, t, flag, inlier, outlier] = scale_propagation( obj, R, t )

flag = true;
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
		obj.params.ransacCoef_scale_prop.weight = atan(-points(3,:)+3)+pi/2;
		[scale, inlier, outlier] = ransac(P1_exp(1:3,:), P1_ini(1:3,:), obj.params.ransacCoef_scale_prop);
		obj.nFeatureInlier = length(inlier);
		
	end
	
	% Use the previous scale, if the scale cannot be found
	if nPoint <= obj.params.ransacCoef_scale_prop.minPtNum || length(inlier) < obj.params.thInlier || isempty(scale)
		warning('there are a few SCALE FACTOR INLIERS')
		scale = norm(obj.TRec{obj.step-1}(1:3,4));
		flag = false;
	end
	
end

%% INITIALIZATION
% Initialze scale, in the case of the first time
if ~obj.scale_initialized
	
	scale = obj.params.initScale;
	obj.nFeatureInlier = obj.nFeature3DReconstructed;
	inlier = find([obj.features(:).is_3D_reconstructed] == true);
	
end

%% UPDATE
% Update scale
t = scale * t;

end