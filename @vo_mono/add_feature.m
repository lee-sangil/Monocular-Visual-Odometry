function obj = add_feature( obj )

% Load bucket parameters
bkSize = obj.bucket.size;
bkSafety = obj.bucket.safety;

% Choose ROI based on the probabilistic approaches with the mass of bucket
[i, j] = idx_randselect(obj.bucket);
ROI = floor([(i-1)*bkSize(1)+1, (j-1)*bkSize(2)+1, bkSize(1), bkSize(2)]);

ROI(1) = max(bkSafety, ROI(1));
ROI(2) = max(bkSafety, ROI(2));
ROI(3) = min(obj.params.imSize(1)-bkSafety, ROI(1)+ROI(3))-ROI(1);
ROI(4) = min(obj.params.imSize(2)-bkSafety, ROI(2)+ROI(4))-ROI(2);

% Seek index of which feature is extracted specific bucket
try
	idx = find(all(bsxfun(@eq, reshape([obj.features(:).bucket], 2, []).', [i, j]), 2) == true);
	nInBucket = length(idx);
catch
	idx = [];
	nInBucket = 0;
end

% Try to find a seperate feature
dist = zeros(1,nInBucket);
filterSize = 5;

while true

	if filterSize < 3
		return;
	end

	crop_image = obj.cur_image(ROI(2):ROI(2)+ROI(4), ROI(1):ROI(1)+ROI(3));

% 	keypoints = cv.FAST(crop_image);
% 	locs = reshape([keypoints.pt], 2, []).';
% 	[~, sidx] = sort([keypoints.response], 2, 'descend');
% 	locs = locs(sidx, :);
	
% 	keypoints = detectKAZEFeatures(crop_image);
% 	locs = keypoints.Location;
	
	keypoints = cv.goodFeaturesToTrack(crop_image, 'UseHarrisDetector', true);
	keypoints = cellfun(@transpose, keypoints, 'un', 0);
	locs = [keypoints{:}].'; % uv-coordinates
	
	if isempty(locs)
		return;
	else
		locs(:,1) = locs(:,1) + ROI(1) - 1;
		locs(:,2) = locs(:,2) + ROI(2) - 1;
	end
	
	for l = 1:size(locs,1)
		for f = 1:length(idx)
			dist(f) = norm(locs(l,:).' - obj.features(idx(f)).uv(:,1));
		end
		if all(dist > 2)
			break;
		end
	end
	if all(dist > 2)
		break;
	end
	
	filterSize = filterSize - 2;
end

% Add new feature to VO object
new = obj.nFeature + 1;

obj.features(new).id = obj.new_feature_id; % unique id of the feature
obj.features(new).uv = double(locs(l,:)).'; % uv point in pixel coordinates
obj.features(new).life = 1; % the number of frames in where the feature is observed
obj.features(new).bucket = [i, j]; % the location of bucket where the feature belong to
obj.features(new).point = nan(4,1); % 3-dim homogeneous point in the local coordinates
obj.features(new).is_matched = false; % matched between both frame
obj.features(new).is_2D_inliered = false; % belong to major (or meaningful) movement
obj.features(new).is_3D_reconstructed = false; % triangulation completion
obj.features(new).is_3D_init = false; % scale-compensated
obj.features(new).point_init = nan(4,1); % scale-compensated 3-dim homogeneous point in the global coordinates
obj.features(new).point_var = obj.params.var_point;

obj.new_feature_id = obj.new_feature_id + 1;
obj.nFeature = obj.nFeature + 1;

% Update bucket
obj.bucket.prob = conv2(obj.bucket.mass, obj.params.kernel, 'same');
obj.bucket.mass(i, j) = obj.bucket.mass(i, j) + 1;

end

function [i, j] = idx_randselect(bucket)

% Calculate weight
weight = 0.05 ./ (bucket.prob(:) + 0.05);

% Select index
idx = 0:numel(bucket.mass)-1;
select = idx( sum( bsxfun(@ge, rand(1), cumsum(weight./sum(weight)))) + 1 );
i = floor(select / bucket.grid(2))+1;
j = mod(select, bucket.grid(2))+1;

end