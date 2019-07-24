function obj = update3DPoints( obj, R, t, inlier, outlier )

if nargin == 5
	%% without-PnP
	scale = norm(t);
	
	if obj.scale_initialized
		
		% Seek index of which feature is 3D reconstructed currently,
		% and 3D initialized previously
		[idx, nPoint] = seek_index(obj, obj.nFeature3DReconstructed, ...
			[obj.features(:).is_3D_reconstructed] & ...
			[obj.features(:).is_3D_init]);
		
		% Get expected 3D point by transforming the coordinates of the
		% observed 3d point
		P1_exp = zeros(4, nPoint);
		for i = 1:nPoint
			P1_exp(:,i) = obj.features(idx(i)).point;
		end
		
		% Update features
		for i = inlier
			% Update 3D point and covariance
			var = (P1_exp(3,i) * scale / 2)^4 * obj.params.var_theta;
			new_var = 1 / ( (1 / obj.features(idx(i)).point_var) + (1 / var) );
			obj.features(idx(i)).point_prev_var = obj.features(idx(i)).point_var;
			obj.features(idx(i)).point_init = new_var/var * obj.TocRec{obj.step-1} * P1_exp(:,i) * scale + new_var/obj.features(idx(i)).point_var * obj.features(idx(i)).point_init;
			obj.features(idx(i)).point_init(4) = 1;
			obj.features(idx(i)).point_var = new_var;
		end
		
		for i = outlier
			% Update life
			obj.features(idx(i)).life = 0;
		end
		
% 		figure(2);
% 		p = obj.TocRec{obj.step-1} \ [obj.features(idx).point_init];
% 		p_k = [obj.features(idx).point]*scale;
% 		scatter3(p(1,:), p(2,:), p(3,:), 'ro');hold on;
% 		scatter3(p_k(1,:), p_k(2,:), p_k(3,:), 'b.');hold off;
% 		xlim([-30 30]);
% 		ylim([-15 2]);
% 		zlim([00 60]);
% 		set(gca,'view',[370,-28]);
	end
	
	% Seek index of which feature is 3D reconstructed currently
	idx = seek_index(obj, obj.nFeature3DReconstructed, [obj.features(:).is_3D_reconstructed]);
	
	% Update features which 3D point is intialized currently
	for i = idx
		obj.features(i).point(1:3,:) = scale * obj.features(i).point(1:3,:);
		
		% Initialize 3D point of each features when it is initialized for
		% the first time
		if ~obj.features(i).is_3D_init
			obj.features(i).point_init = obj.TocRec{obj.step-1} * obj.features(i).point;
			obj.features(i).is_3D_init = true;
		end
	end
	
elseif nargin == 4
	%% with-PnP
	% Load parameters from object
	K = obj.params.K;
	nFeature2DInliered = obj.nFeature2DInliered;
	
	% Extract homogeneous 2D point which is inliered with essential constraint
	idx = seek_index(obj, obj.nFeature2DInliered, [obj.features(:).is_2D_inliered]);
	
	uv_prev = zeros(2, nFeature2DInliered);
	uv_curr = zeros(2, nFeature2DInliered);
	for i = 1:length(idx)
		uv_prev(:,i) = obj.features(idx(i)).uv(:,2);
		uv_curr(:,i) = obj.features(idx(i)).uv(:,1);
	end
	x_prev = K \ [uv_prev; ones(1, nFeature2DInliered)];
	x_curr = K \ [uv_curr; ones(1, nFeature2DInliered)];
	
	[X_prev, X_curr, lambda_prev, lambda_curr] = obj.contructDepth(x_prev, x_curr, R, t);
	inliers = find(lambda_prev > 0 & lambda_curr > 0);
	point_prev = [X_prev(:,inliers);ones(1,length(inliers))];
	
	idx = idx(inliers);
	for i = 1:length(idx)
		% Update 3D point of features with respect to the current image
		obj.features(idx(i)).point = point_prev(:,i);
		obj.features(idx(i)).is_3D_reconstructed = true;
	end
	obj.nFeature3DReconstructed = length(idx);
	
	idx = seek_index(obj, obj.nFeature3DReconstructed, ...
			[obj.features(:).is_3D_reconstructed] & ...
			[obj.features(:).is_3D_init]);
		
	% Update features
	for i = inlier
		% Update 3D point and covariance
		var = (obj.features(idx(i)).point(3) / 2)^4 * obj.params.var_theta;
		new_var = 1 / ( (1 / obj.features(idx(i)).point_var) + (1 / var) );
		obj.features(idx(i)).point_prev_var = obj.features(idx(i)).point_var;
		obj.features(idx(i)).point_init = new_var/var * obj.TocRec{obj.step-1} * obj.features(idx(i)).point + new_var/obj.features(idx(i)).point_var * obj.features(idx(i)).point_init;
		obj.features(idx(i)).point_init(4) = 1;
		obj.features(idx(i)).point_var = new_var;
	end
	
	% Seek index of which feature is 3D reconstructed currently
	idx = seek_index(obj, obj.nFeature3DReconstructed, [obj.features(:).is_3D_reconstructed]);
	
	% Update features which 3D point is intialized currently
	for i = idx
		% Initialize 3D point of each features when it is initialized for
		% the first time
		if ~obj.features(i).is_3D_init
			obj.features(i).point_init = obj.TocRec{obj.step-1} * obj.features(i).point;
			obj.features(i).is_3D_init = true;
		end
	end
end