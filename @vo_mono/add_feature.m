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

	[des, locs] = descriptor(obj.cur_image, 'ROI', ROI, 'FilterSize', filterSize);
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

end