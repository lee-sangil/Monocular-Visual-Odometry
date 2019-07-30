function [T, Toc, Poc] = update3DPoints( obj, R, t, inlier, outlier, mode, R_E, t_E, success_E )

if strcmp(mode, 'w/oPnP')
	
	%% without-PnP
	scale = norm(t);
	
	% Seek index of which feature is 3D reconstructed currently,
	% and 3D initialized previously
	idx3D = find([obj.features(:).is_3D_reconstructed] == true);
	nPoint = length(idx3D);
	
	if obj.scale_initialized
		% Get expected 3D point by transforming the coordinates of the
		% observed 3d point
		P1_exp = zeros(4, nPoint);
		for i = 1:nPoint
			P1_exp(:,i) = obj.features(idx3D(i)).point;
		end
		
		for i = outlier
			% Update life
			obj.features(i).life = 0;
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
	
	% Update features which 3D point is intialized currently
	for i = 1:nPoint
		obj.features(idx3D(i)).point(1:3,:) = scale * obj.features(idx3D(i)).point(1:3,:);
		
		% Initialize 3D point of each features when it is initialized for
		% the first time
		if ~obj.features(idx3D(i)).is_3D_init
			obj.features(idx3D(i)).point_init = obj.TocRec{obj.step-1} * obj.features(idx3D(i)).point;
			obj.features(idx3D(i)).is_3D_init = true;
		else
			% Update 3D point and covariance
			var = (P1_exp(3,i) * scale / 2)^4 * obj.params.var_theta;
			new_var = 1 / ( (1 / obj.features(idx3D(i)).point_var) + (1 / var) );
			obj.features(idx3D(i)).point_init = new_var/var * obj.TocRec{obj.step-1} * P1_exp(:,i) * scale + new_var/obj.features(idx3D(i)).point_var * obj.features(idx3D(i)).point_init;
			obj.features(idx3D(i)).point_init(4) = 1;
			obj.features(idx3D(i)).point_var = new_var;
		end
	end
	
	T = [R' -R'*t; 0 0 0 1];
	Toc = obj.TocRec{obj.step-1} * T;
	Poc = Toc * [0 0 0 1]';
	
elseif strcmp(mode, 'w/PnP')
	%% with-PnP
	% Load parameters from object
	K = obj.params.K;
	
	% Extract homogeneous 2D point which is inliered with essential constraint
	idx3D = find([obj.features(:).is_3D_init] == true);
	nPoint = length(idx3D);
	
	x_init_ = zeros(3, nPoint);
	for i = 1:length(idx3D)
		x_init_(:,i) = obj.features(idx3D(i)).point_init(1:3);
	end
	
	Toc = [R' -R'*t; 0 0 0 1];
	T = obj.TocRec{obj.step-1} \ Toc;
	Poc = Toc * [0 0 0 1]';
	
	Tco = [R t; 0 0 0 1];
	point_curr = Tco * [x_init_; ones(1,length(x_init_))];
	
	%% Update 3D points in local coordinates
	scale = norm(T(1:3,4));
	point_prev = T * point_curr;
	
% 	idx3D = idx3D(inliers);
	for i = 1:length(idx3D)
		% Update 3D point of features with respect to the current image
		obj.features(idx3D(i)).point = point_prev(:,i);
		obj.features(idx3D(i)).is_3D_reconstructed = true;
	end
	
	%% Initialize 3D points in global coordinates
	% Extract homogeneous 2D point which is inliered with essential constraint
	idx2D = find([obj.features(:).is_2D_inliered] == true);
	
	if success_E
		Rinv = R_E;
		tinv = t_E;
% 		idx2D = idx2D(inlier);
		
% 		for i = outlier
% 			% Update life
% 			obj.features(i).life = 0;
% 		end
	else
		Rinv = T(1:3,1:3).';
		tinv = -T(1:3,1:3).'*T(1:3,4);
	end
	
	nPoint = length(idx2D);
	
	uv_prev = zeros(2, nPoint);
	uv_curr = zeros(2, nPoint);
	for i = 1:length(idx2D)
		uv_prev(:,i) = obj.features(idx2D(i)).uv(:,2);
		uv_curr(:,i) = obj.features(idx2D(i)).uv(:,1);
	end
	x_prev = K \ [uv_prev; ones(1, nPoint)];
	x_curr = K \ [uv_curr; ones(1, nPoint)];
	
	[X_prev, X_curr, lambda_prev, lambda_curr] = obj.contructDepth(x_prev, x_curr, Rinv, tinv);
	inliers = find(lambda_prev > 0 & lambda_curr > 0);
	point_prev = [X_prev(:,inliers);ones(1,length(inliers))];
	point_curr = [X_curr(:,inliers);ones(1,length(inliers))];
	
	idx2DInlier = idx2D(inliers);
	for i = 1:length(idx2DInlier)
		if obj.features(idx2DInlier(i)).is_3D_init == false
			% Update 3D point of features with respect to the current image
			obj.features(idx2DInlier(i)).point = point_prev(:,i);
			obj.features(idx2DInlier(i)).is_3D_reconstructed = true;
			
			obj.features(idx2DInlier(i)).point_init = Toc*point_curr(:,i);
			obj.features(idx2DInlier(i)).is_3D_init = true;
		else
% 			var = 2 * point_curr(3,i) / scale * obj.params.var_theta;
			var = (point_curr(3,i) / scale)^4 * obj.params.var_theta;
			new_var = 1 / ( (1 / obj.features(idx2DInlier(i)).point_var) + (1 / var) );
			obj.features(idx2DInlier(i)).point_init = new_var/var * Toc * point_curr(:,i) * scale + new_var/obj.features(idx2DInlier(i)).point_var * obj.features(idx2DInlier(i)).point_init;
			obj.features(idx2DInlier(i)).point_init(4) = 1;
			obj.features(idx2DInlier(i)).point_var = new_var;
		end
	end
	
	 idx = find([obj.features(:).is_3D_reconstructed] == true);
	 obj.nFeature3DReconstructed = length(idx);
end