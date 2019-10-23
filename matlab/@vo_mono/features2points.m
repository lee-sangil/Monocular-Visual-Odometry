function points = features2points(obj)

% constant
features = obj.features;

points = zeros(obj.nFeature, 2);
for i = 1:obj.nFeature
	points(i,:) = features(i).uv2';
end