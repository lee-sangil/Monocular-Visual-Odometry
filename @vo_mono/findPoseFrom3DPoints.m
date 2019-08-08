function [R, t, flag, inlier, outlier] = findPoseFrom3DPoints( obj )

% Seek index of which feature is 3D reconstructed currently,
% and 3D initialized previously
idx = find([obj.features(:).is_3D_init] == true & ...
	[obj.features(:).life] > 10 );
% 	[obj.features(:).point_var] < 0.1 & ... % Theoretically
nPoint = length(idx);

% Use RANSAC to find suitable scale
if nPoint > obj.params.thInlier
	objectPoints_ = [obj.features(idx).point_init];
	objectPoints = permute(objectPoints_(1:3,:),[3 2 1]);
	
	imagePoints_ = zeros(2,nPoint);
	for i = 1:nPoint
		imagePoints_(:,i) = obj.features(idx(i)).uv(:,1);
	end
	imagePoints = permute(imagePoints_,[3 2 1]);
	
	[r_vec, t_vec, success, inlier] = cv.solvePnPRansac(objectPoints, imagePoints, obj.params.K, 'IterationsCount', 1e4, 'ReprojectionError', obj.params.reprojError);
	if success == false
		[r_vec, t_vec, success, inlier] = cv.solvePnPRansac(objectPoints, imagePoints, obj.params.K, 'IterationsCount', 1e4, 'ReprojectionError', obj.params.reprojError*2);
	end
	
	R = expm(skew(r_vec));
	t = t_vec;
	flag = success;
	inlier = idx(transpose(inlier) + 1);
	outlier = idx(~ismember(idx,inlier));
else
	R = [];
	t = [];
	inlier = [];
	outlier = [];
	flag = false;
end

end