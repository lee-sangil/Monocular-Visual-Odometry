function flag = calculate_essential(obj)

K = obj.params.K;

if obj.step == 1
	flag = true;
	return;
end

[idx, nInlier] = seek_index(obj, obj.nFeatureMatched, [obj.features(:).is_matched]);

uv1 = zeros(2, nInlier);
uv2 = zeros(2, nInlier);
for i = 1:nInlier
	uv1(:,i) = obj.features(idx(i)).uv1;
	uv2(:,i) = obj.features(idx(i)).uv2;
end
x1 = K \ [uv1; ones(1, nInlier)];
x2 = K \ [uv2; ones(1, nInlier)];

ransacCoef.iterMax = 1e4;
ransacCoef.minPtNum = 8;
ransacCoef.thInlrRatio = 0.99;
ransacCoef.thDist = 1e-5;
ransacCoef.thDistOut = 1e-5;
ransacCoef.funcFindF = @(x1, x2) obj.eight_point_essential_hypothesis(x1, x2);
ransacCoef.funcDist = @(hyp, x1, x2) obj.essential_model_error(hyp, x1, x2);

[E, inlier, ~] = ransac(x1, x2, ransacCoef);

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
	