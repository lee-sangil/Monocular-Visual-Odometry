function [R, t] = findPoseFrom3DPoints( obj )
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
	objectPoints = [obj.features(:).point];
	imagePoints = [obj.features(:).uv(:,1)];
	[R, t, success, inliers] = cv.solvePnPRansac(objectPoints, imagePoints, obj.params.K);
end

end