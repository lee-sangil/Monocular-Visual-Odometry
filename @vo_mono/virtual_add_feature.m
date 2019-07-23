function obj = virtual_add_feature( obj, features_, points_, id_ )

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
	[idx, nInBucket] = seek_index(obj, obj.nFeatureMatched, all(bsxfun(@eq, reshape([obj.features(:).bucket], 2, []).', [i, j]), 2));
catch
	idx = [];
	nInBucket = 0;
end

% Try to find a seperate feature
dist = zeros(1,nInBucket);

valid_roi = features_(1,:) >= ROI(1) & features_(1,:) <= ROI(1)+ROI(3) & features_(2,:) >= ROI(2) & features_(2,:) <= ROI(2)+ROI(4);
locs = features_(:,valid_roi)';
ids = id_(valid_roi);

if isempty(locs)
	return;
end

flag = false;
for l = 1:size(locs,1)
	for f = 1:length(idx)
		dist(f) = norm(locs(l,:).' - obj.features(idx(f)).uv(:,1));
	end
	if all(dist > 2)
		break;
	end
end
if all(dist > 2)
	flag = true;
end
if ~isempty(dist) & all(dist==0)
	return;
end

if flag == false
	return;
end

% Add new feature to VO object
new = obj.nFeature + 1;

obj.features(new).id = ids(l);
obj.features(new).uv = double(locs(l,:)).';
obj.features(new).life = 1;
obj.features(new).bucket = [i, j];
obj.features(new).point = nan(4,1);
obj.features(new).is_matched = false;
obj.features(new).is_2D_inliered = false;
obj.features(new).is_3D_reconstructed = false;
obj.features(new).is_3D_init = false;
obj.features(new).point_init = nan(4,1);
obj.features(new).point_var = obj.params.var_point;
obj.features(new).point_prev_var = obj.params.var_point;

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