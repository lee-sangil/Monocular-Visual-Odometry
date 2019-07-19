function [R, t] = scale_propagation( obj, R, t )

%% scale propagation
[idx, nPoint] = seek_index(obj, obj.nFeature3DReconstructed, ...
							[obj.features(:).is_3D_reconstructed] & ...
							[obj.features(:).is_3D_init] & ...
							[obj.features(:).point_init_step] < obj.step);

if ~isempty(idx)
	
	P1_ref = zeros(3, nPoint);
	P1 = zeros(3, nPoint);
	for i = 1:nPoint
		T = obj.features(i).transform_to_step;
		P1_ref(:,i) = T(1:3,:) * obj.features(idx(i)).point_init;
	end
	for i = 1:nPoint
		P1(:,i) = obj.features(idx(i)).point(1:3);
	end

	ransacCoef.iterMax = 1e3;
	ransacCoef.minPtNum = 1;
	ransacCoef.thInlrRatio = 0.99;
	ransacCoef.thDist = 2;
	ransacCoef.thDistOut = 5;
	ransacCoef.funcFindF = @(x1, x2) obj.calculate_scale(x1, x2);
	ransacCoef.funcDist = @(hyp, x1, x2) obj.calculate_scale_error(hyp, x1, x2);
	
	[scale, inlier, ~] = ransac(P1, P1_ref, ransacCoef);
	if isempty(scale)
		scale = norm(obj.TRec{obj.step-1}(1:3,4));
	else
		obj.nFeatureInlier = length(inlier);
	end
	
	t = t * scale;

end

end