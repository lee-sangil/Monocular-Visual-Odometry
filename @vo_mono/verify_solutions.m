function [R, t] = verify_solutions( obj, R_vec, t_vec )
%% VERIFICATION
% Load parameters from object
K = obj.params.K;
nFeature2DInliered = obj.nFeature2DInliered;

% Extract homogeneous 2D point which is inliered with essential constraint
idx = find([obj.features(:).is_2D_inliered] == true);

uv_prev = zeros(2, nFeature2DInliered);
uv_curr = zeros(2, nFeature2DInliered);
for i = 1:length(idx)
	uv_prev(:,i) = obj.features(idx(i)).uv(:,2);
	uv_curr(:,i) = obj.features(idx(i)).uv(:,1);
end
x_prev = K \ [uv_prev; ones(1, nFeature2DInliered)];
x_curr = K \ [uv_curr; ones(1, nFeature2DInliered)];

% Find reasonable rotation and translational vector
max_num = 0;
for i = 1:length(R_vec)
	R1 = R_vec{i};
	t1 = t_vec{i};
	
	[X_prev, ~, lambda_prev, lambda_curr] = obj.contructDepth(x_prev, x_curr, R1, t1);
	inlier = find(lambda_prev > 0 & lambda_curr > 0);
	
	if length(inlier) > max_num
		max_num = length(inlier);
		max_inlier = inlier;
		point_prev = [X_prev(:,inlier);ones(1,length(inlier))];
		
		R = R1;
		t = t1;
	end
end

%% STORE
% Store 3D characteristics in features
if max_num < nFeature2DInliered*0.5
	R = [];
	t = [];
	
else
	idx = idx(max_inlier);
	for i = 1:length(idx)
		% Update 3D point of features with respect to the current frame
		obj.features(idx(i)).point = point_prev(:,i);
		obj.features(idx(i)).is_3D_reconstructed = true;
	end
	obj.nFeature3DReconstructed = length(idx);
	
end