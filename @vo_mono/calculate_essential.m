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
x_prev = K \ [uv_prev; ones(1, nInlier)];
x_curr = K \ [uv_curr; ones(1, nInlier)];

% RANSAC
% 	obj.params.ransacCoef_calc_essential.weight = ones(length(idx),1);
obj.params.ransacCoef_calc_essential.weight = [obj.features(idx).life].';
[E, inlier, ~] = ransac(x_prev, x_curr, obj.params.ransacCoef_calc_essential);
% cv.findEssentialMat()
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
idx = idx(inlier);
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
	