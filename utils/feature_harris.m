function [des, locs] = feature_harris(image, varargin)

if size(image, 3) == 3
	image = rgb2gray(image);
end

regions = detectHarrisFeatures(image, varargin{:});
locs = regions.Location(:, [2 1]);
des = extract_pattern(image, round(locs));

% [descriptors, valid_points] = extractFeatures(image, regions, 'Upright', true);
% locs = valid_points.Location(:, [2 1]);
% des = bsxfun(@rdivide, double(descriptors.Features), sum(double(descriptors.Features).^2, 2).^(1/2));