function obj = update_bucket( obj )

nFeature = obj.nFeature;

obj.bucket.mass = zeros(obj.bucket.grid);
for i = 1:nFeature
	uv = obj.features(i).uv(:,1).';
	idx_bucket = ceil(uv./obj.params.imSize.*obj.bucket.grid);
	obj.features(i).bucket = idx_bucket;
	obj.bucket.mass(idx_bucket(1), idx_bucket(2)) = obj.bucket.mass(idx_bucket(1), idx_bucket(2)) + 1;
end