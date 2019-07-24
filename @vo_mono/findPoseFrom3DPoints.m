function [R, t, flag] = findPoseFrom3DPoints( obj )
%% TRI-IMAGES SCALE COMPENSATION

% Seek index of which feature is 3D reconstructed currently,
% and 3D initialized previously
[idx, nPoint] = seek_index(obj, obj.nFeature3DReconstructed, ...
	[obj.features(:).is_3D_reconstructed] ...
	& [obj.features(:).is_3D_init] ...
	& [obj.features(:).life] > 10 ... % Temporary
	);
% 	& [obj.features(:).point_var] < 0.1 ... % Theoretically

% Use RANSAC to find suitable scale
if nPoint > obj.params.thInlier
	objectPoints_ = [obj.features(idx).point_init];
	objectPoints = permute(objectPoints_(1:3,:),[3 2 1]);
	
	imagePoints_ = zeros(2,nPoint);
	for i = 1:nPoint
		imagePoints_(:,idx(i)) = obj.features(idx(i)).uv(:,1);
	end
	imagePoints = permute(imagePoints_,[3 2 1]);
	[R, t, success, inliers] = cv.solvePnPRansac(objectPoints, imagePoints, obj.params.K);
	flag = success;
else
	R = [];
	t = [];
	flag = false;
end

end