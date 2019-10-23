<<<<<<< HEAD:@vo_mono/add_feature.m
function obj = add_feature( obj )

% variable
bucketMass = obj.bucket.mass;

% constant
bucketSize = obj.bucket.size;
bucketGrid = obj.bucket.grid;
descriptor = obj.params.descriptor;

%
[i, j] = idx_randselect(bucketMass, bucketGrid, obj.params.kernel);
ROI = floor([(i-1)*bucketSize(1)+1, (j-1)*bucketSize(2)+1, bucketSize(1), bucketSize(2)]);

ROI(1) = max(20, ROI(1));
ROI(2) = max(20, ROI(2));
ROI(3) = min(size(obj.cur_image, 2)-20, ROI(1)+ROI(3))-ROI(1);
ROI(4) = min(size(obj.cur_image, 1)-20, ROI(2)+ROI(4))-ROI(2);

locs = [];
dist = zeros(1,obj.nFeature);
filterSize = 5;

while true

	if filterSize < 3
		return;
	end

	[des, locs] = descriptor(obj.cur_image, 'ROI', ROI);
	if isempty(locs)
		return;
	end
	
	for l = 1:size(locs,1)
		for f = 1:obj.nFeature
			dist(f) = norm(locs(l,[2 1]).' - obj.features(f).uv2);
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

% add new feature
new = obj.nFeature + 1;

obj.features(new).id = obj.new_feature_id;
obj.features(new).uv1 = nan(1,2);
obj.features(new).uv2 = locs(l,[2 1]).';
obj.features(new).uvs = locs(l,[2 1]).';
obj.features(new).life = 1;
obj.features(new).des = des(l,:).';
obj.features(new).bucket = [i, j];
obj.features(new).point = nan(4,1);
obj.features(new).is_matched = false;
obj.features(new).is_2D_inliered = false;
obj.features(new).is_3D_reconstructed = false;
obj.features(new).is_3D_init = false;
obj.features(new).point_init_step = 0;
obj.features(new).point_init = nan(4,1);
obj.features(new).transform_from_init = eye(4);
obj.features(new).transform_to_step = eye(4);

obj.new_feature_id = obj.new_feature_id + 1;
obj.nFeature = obj.nFeature + 1;

% update bucket mass
bucketMass(i, j) = bucketMass(i, j) + 1;
obj.bucket.mass = bucketMass;

end

function [i, j] = idx_randselect(bucketMass, bucketGrid, kernel)

bucketProb = conv2(bucketMass, kernel, 'same');
weight = 0.05 ./ (bucketProb(:) + 0.05);
% weight = 1 ./ (bucketMass(:) + 0.5).^2;
% weight = ones(prod(bucketGrid),1);

idx = 0:numel(bucketMass)-1;
select = idx( sum( bsxfun(@ge, rand(1), cumsum(weight./sum(weight)))) + 1 );
i = floor(select / bucketGrid(2))+1;
j = mod(select, bucketGrid(2))+1;

=======
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
		if all(dist > obj.params.min_px_dist)
			break;
		end
	end
	if all(dist > obj.params.min_px_dist)
		break;
	end
	
	filterSize = filterSize - 2;
end

% Add new feature to VO object
new = obj.nFeature + 1;

obj.features(new).id = obj.new_feature_id; % unique id of the feature
obj.features(new).frame_init = obj.step; % frame step when the feature is created
obj.features(new).uv = double(locs(l,:)).'; % uv point in pixel coordinates
obj.features(new).life = 1; % the number of frames in where the feature is observed
obj.features(new).bucket = [i, j]; % the location of bucket where the feature belong to
obj.features(new).point = nan(4,1); % 3-dim homogeneous point in the local coordinates
obj.features(new).is_matched = false; % matched between both frame
obj.features(new).is_wide = false; % verify whether features btw the initial and current are wide enough
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

>>>>>>> debug:matlab/@vo_mono/add_feature.m
end