function flag = calculate_essential(obj)
%% CALCULATION
% Load camera intrinsic matrix
K = obj.params.K;

if obj.step == 1
	flag = true;
	return;
end

% Extract homogeneous 2D point which is matched with corresponding feature
[idx, nInlier] = seek_index(obj, obj.nFeatureMatched, [obj.features(:).is_matched]);

uv1 = zeros(2, nInlier); % previous image plane
uv2 = zeros(2, nInlier); % current image plane
for i = 1:nInlier
	uv1(:,i) = obj.features(idx(i)).uv(:,2);
	uv2(:,i) = obj.features(idx(i)).uv(:,1);
end
x1 = K \ [uv1; ones(1, nInlier)];
x2 = K \ [uv2; ones(1, nInlier)];

% RANSAC
obj.params.ransacCoef_calc_essential.weight = [obj.features(idx).life].';
[E, inlier, ~] = ransac(x1, x2, obj.params.ransacCoef_calc_essential);

%% STORE
idx = idx(inlier);
for i = idx
	obj.features(i).is_2D_inliered = true;
end
obj.nFeature2DInliered = length(idx);

if obj.nFeature2DInliered < obj.params.thInlier
	warning('there are a few 2D PIXEL INLIERS');
	flag = false;
else
	obj.essential = E;
	flag = true;
end
	