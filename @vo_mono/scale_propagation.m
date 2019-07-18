function [R, t] = scale_propagation( obj, R, t )
%% TRI-IMAGES SCALE COMPENSATION
if obj.scale_initialized
	
	% Seek index of which feature is 3D reconstructed currently,
	% and 3D initialized previously
	[idx, nPoint] = seek_index(obj, obj.nFeature3DReconstructed, ...
									[obj.features(:).is_3D_reconstructed] & ...
									[obj.features(:).is_3D_init]);
	
	% Use RANSAC to find suitable scale
	if nPoint > obj.params.ransacCoef_scale_prop.minPtNum
		
		% Get initialized 3D point
		P1_ini = zeros(4, nPoint);
		for i = 1:nPoint
			P1_ini(:,i) = obj.TocRec{obj.step-1} * obj.features(idx(i)).point_init;
		end
		
		% Get expected 3D point by transforming the coordinates of the
		% observed 3d point
		P1_exp = zeros(4, nPoint);
		for i = 1:nPoint
			P1_exp(:,i) = obj.features(idx(i)).point;
		end
		
		% RANSAC
		obj.params.ransacCoef_scale_prop.weight = [obj.features(idx).life] ./ sqrt([obj.features(idx).point_var]);
		[scale, inlier, outlier] = ransac(P1_exp(1:3,:), P1_ini(1:3,:), obj.params.ransacCoef_scale_prop);
		obj.nFeatureInlier = length(inlier);
		
		% Update features
		for i = inlier
			% Update 3D point and covariance
			var = (P1_exp(3,i) * scale / 2)^4 * obj.params.var_theta;
			new_var = 1 / ( (1 / obj.features(idx(i)).point_var) + (1 / var) );
			obj.features(idx(i)).point_prev_var = obj.features(idx(i)).point_var;
			obj.features(idx(i)).point_init = new_var/var * P1_exp(:,i) * scale + new_var/obj.features(idx(i)).point_var * obj.features(idx(i)).point_init;
			obj.features(idx(i)).point_init(4) = 1;
			obj.features(idx(i)).point_var = new_var;
		end
		
		for i = outlier
			% Update life
			obj.features(idx(i)).life = 0;
		end
	end
	
	% Use the previous scale, if the scale cannot be found
	if nPoint <= obj.params.ransacCoef_scale_prop.minPtNum || length(inlier) < obj.params.thInlier || isempty(scale)
		warning('there are a few SCALE FACTOR INLIERS')
		scale = norm(obj.TRec{obj.step-1}(1:3,4));
	end
	
end

%% INITIALIZATION
% Seek index of which feature is 3D reconstructed currently
idx = seek_index(obj, obj.nFeature3DReconstructed, [obj.features(:).is_3D_reconstructed]);

% Initialze scale, in the case of the first time
if ~obj.scale_initialized
	
	scale = .1;
% 	scale = 10 / ( [0 0 1 0] * min([obj.features(idx).point],[],2) );
	obj.nFeatureInlier = length(idx);
	obj.scale_initialized = true;
	
end

%% UPDATE
% Update scale
t = scale * t;

% Update features which 3D point is intialized currently
for i = idx
	obj.features(i).point(1:3,:) = scale * obj.features(i).point(1:3,:);

	% Initialize 3D point of each features when it is initialized for
	% the first time
	if ~obj.features(i).is_3D_init
		obj.features(i).point_init = obj.TocRec{obj.step-1} \ obj.features(i).point;
		obj.features(i).is_3D_init = true;
	end
end

% disp(scale);
% disp( norm(obj.TRec{obj.step-1}(1:3,4)) );

end