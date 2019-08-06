function obj = set_image( obj, image )

if ismember(obj.step, obj.params.bootstrap.frames) || obj.initialized
	if size(image,3) == 3
		image = rgb2gray(image);
	end
	obj.prev_image = obj.curr_image;
	obj.undist_image = cv.undistort(image, obj.params.intrinsic, [obj.params.radialDistortion(1:2) obj.params.tangentialDistortion]);
	obj.curr_image = obj.undist_image;
end