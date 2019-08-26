function obj = set_image( obj, image )

obj.prev_image = obj.cur_image;
obj.undist_image = cv.undistort(image, obj.params.K, [obj.params.radialDistortion(1:2) obj.params.tangentialDistortion]);
obj.cur_image = obj.undist_image;