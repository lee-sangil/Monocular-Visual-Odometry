function [des, locs] = feature_surf(image, varargin)

if size(image, 3) == 3
	image = rgb2gray(image);
end

regions = detectSURFFeatures(image, varargin{:});
[des, valid_points] = extractFeatures(image, regions, 'Upright', true);
locs = valid_points.Location(:, [2 1]);