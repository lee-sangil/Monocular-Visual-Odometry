function flag = calculate_essential(obj)
%% CALCULATION
% Load camera intrinsic matrix
K = obj.params.K;

if obj.step == 1
	flag = true;
	return;
end

% Extract homogeneous 2D point which is matched with corresponding feature
idx = find([obj.features(:).is_matched]==true);
nInlier = length(idx);

uv_prev = zeros(2, nInlier); % previous image plane
uv_curr = zeros(2, nInlier); % current image plane
for i = 1:nInlier
	uv_prev(:,i) = obj.features(idx(i)).uv(:,2); % second-to-latest
	uv_curr(:,i) = obj.features(idx(i)).uv(:,1); % latest
end

% RANSAC
uv_prev_cell = mat2cell(uv_prev.', ones(1,size(uv_prev,2)));
uv_curr_cell = mat2cell(uv_curr.', ones(1,size(uv_curr,2)));
[E, inlier] = cv.findEssentialMat(uv_prev_cell, uv_curr_cell, 'Method', 'Ransac', 'CameraMatrix', K, 'Threshold', 1e-1);

[U, ~, V] = svd(E);
if det(U) <  0
	U(:,3) = -U(:,3);
end
if det(V) < 0
	V(:,3) = -V(:,3);
end

% Extract rotational and translational movement
W = [0 -1 0; 1 0 0; 0 0 1];
R_vec{1} = U*W*V.';
R_vec{2} = U*W*V.';
R_vec{3} = U*W.'*V.';
R_vec{4} = U*W.'*V.';
t_vec{1} = U(:,3);
t_vec{2} = -U(:,3);
t_vec{3} = U(:,3);
t_vec{4} = -U(:,3);

%% STORE
idx = idx(inlier==1);
for i = idx
	obj.features(i).is_2D_inliered = true;
end
obj.nFeature2DInliered = length(idx);

if obj.nFeature2DInliered < obj.params.thInlier
	warning('there are a few inliers matching features in 2d.');
	flag = false;
else
	obj.R_vec = R_vec;
	obj.t_vec = t_vec;
	flag = true;
end
	