function [T, Toc, Poc] = update3DPoints( obj, R, t, inlier, outlier, mode )

if strcmp(mode, 'w/oPnP')
	%% without-PnP
	scale = norm(t);
	
	% Seek index of which feature is 3D reconstructed currently,
	% and 3D initialized previously
	idx = find([obj.features(:).is_3D_reconstructed] == true);
	nPoint = length(idx);
	
	if obj.scale_initialized
		% Get expected 3D point by transforming the coordinates of the
		% observed 3d point
		P1_exp = zeros(4, nPoint);
		for i = 1:nPoint
			P1_exp(:,i) = obj.features(idx(i)).point;
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
	
	% Update features which 3D point is intialized currently
	for i = 1:nPoint
		obj.features(idx(i)).point(1:3,:) = scale * obj.features(idx(i)).point(1:3,:);
		
		% Initialize 3D point of each features when it is initialized for
		% the first time
		if ~obj.features(idx(i)).is_3D_init
			obj.features(idx(i)).point_init = obj.TocRec{obj.step-1} * obj.features(idx(i)).point;
			obj.features(idx(i)).is_3D_init = true;
		else
			% Update 3D point and covariance
			var = (P1_exp(3,i) * scale / 2)^4 * obj.params.var_theta;
			new_var = 1 / ( (1 / obj.features(idx(i)).point_var) + (1 / var) );
			obj.features(idx(i)).point_init = new_var/var * obj.TocRec{obj.step-1} * P1_exp(:,i) * scale + new_var/obj.features(idx(i)).point_var * obj.features(idx(i)).point_init;
			obj.features(idx(i)).point_init(4) = 1;
			obj.features(idx(i)).point_var = new_var;
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
	idx = find([obj.features(:).is_3D_init] == true);
	nPoint = length(idx);
	
	x_init_ = zeros(3, nPoint);
	uv_curr = zeros(2, nPoint);
	for i = 1:length(idx)
		x_init_(:,i) = obj.features(idx(i)).point_init(1:3);
		uv_curr(:,i) = obj.features(idx(i)).uv(:,1);
	end
	x_init = x_init_ ./ x_init_(3,:);
	x_curr = K \ [uv_curr; ones(1, nPoint)];
	
	[X_init, X_curr, lambda_prev, lambda_curr] = obj.contructDepth(x_init, x_curr, R, t);
	inliers = find(lambda_prev > 0 & lambda_curr > 0);
	point_init = [X_init(:,inliers);ones(1,length(inliers))];
	point_curr = [X_curr(:,inliers);ones(1,length(inliers))];
	
	Toc = [R' -R'*t; 0 0 0 1];
	T = obj.TocRec{obj.step-1} \ Toc;
	Poc = Toc * [0 0 0 1]';
	
	scale = norm(T(1:3,4));
	point_prev = T * point_curr;
	
	idx = idx(inliers);
	for i = 1:length(idx)
		% Update 3D point of features with respect to the current image
		obj.features(idx(i)).point = point_prev(:,i);
		obj.features(idx(i)).is_3D_reconstructed = true;
	end	
	obj.nFeature3DReconstructed = length(idx);
	
end