function [time, image_left, image_right] = get_current_image(obj)

if obj.step < obj.imLength
	obj.step = obj.step + 1;
	
	left_filename = [ obj.imLeftDir sprintf( '%.10d.png', obj.step + obj.imInit - 1) ];
	right_filename = [ obj.imRightDir sprintf( '%.10d.png', obj.step + obj.imInit - 1) ];
	image_left = imread(left_filename);
	image_right = imread(right_filename);
	
% 	if size(image_left, 3) ~= 1
% 		image_left = rgb2gray(image_left);
% 	end
% 	if size(image_right, 3) ~= 1
% 		image_right = rgb2gray(image_right);
% 	end
	
% 	if isa(image_left, 'uint8')
% 		image_left = single(image_left);
% 	end
% 	if isa(image_right, 'uint8')
% 		image_right = single(image_right);
% 	end

% % 	distCoeffs1 = [obj.stereoParams.CameraParameters1.RadialDistortion(1:2), obj.stereoParams.CameraParameters1.TangentialDistortion, obj.stereoParams.CameraParameters1.RadialDistortion(3)];
% % 	distCoeffs2 = [obj.stereoParams.CameraParameters2.RadialDistortion(1:2), obj.stereoParams.CameraParameters2.TangentialDistortion, obj.stereoParams.CameraParameters2.RadialDistortion(3)];
% % 	S = cv.stereoRectify(obj.stereoParams.CameraParameters1.IntrinsicMatrix, distCoeffs1, obj.stereoParams.CameraParameters2.IntrinsicMatrix, distCoeffs2, size(image_left), (obj.stereoParams.RotationOfCamera2), obj.stereoParams.TranslationOfCamera2');
% % 	
% % % 	imgRect_left = cv.undistort(image_left, obj.stereoParams.CameraParameters1.IntrinsicMatrix, distCoeffs1, 'NewCameraMatrix', S.P1);
% % % 	imgRect_right = cv.undistort(image_right, obj.stereoParams.CameraParameters2.IntrinsicMatrix, distCoeffs2, 'NewCameraMatrix', S.P2);
% % 	[map1x, map1y] = cv.initUndistortRectifyMap(obj.stereoParams.CameraParameters1.IntrinsicMatrix, distCoeffs1, size(image_left), 'R', S.R1, 'NewCameraMatrix', S.P1);
% % 	[map2x, map2y] = cv.initUndistortRectifyMap(obj.stereoParams.CameraParameters2.IntrinsicMatrix, distCoeffs2, size(image_left), 'R', S.R2, 'NewCameraMatrix', S.P2);
% % 	imgRect_left = cv.remap(image_left', map1x, map1y);
% % 	imgRect_right = cv.remap(image_right', map2x, map2y);
% 	
% 	imgRect_left = image_left;
% 	imgRect_right = image_right;
	
% 	disparityMap = disparity(imgRect_left, imgRect_right);
% 	depth_left = 0.38757*721.54./disparityMap;
% 	subplot(212);imagesc(depth_left);	

% 	param.disp_min    = 0;           % minimum disparity (positive integer)
% 	param.disp_max    = 255;         % maximum disparity (positive integer)
% 	param.subsampling = false; % process only each 2nd pixel (1=active)
% 	[D1,D2] = elasMex(imgRect_left',imgRect_right',param);
% 	
% 	depth_left = 0.38757*721.54./D1';
	
	% To record depth image for C++ implementation
% 	depth = uint16(depth_left * 1000);
% 	imwrite(depth, sprintf( 'depth/%.10d.png', obj.step + obj.imInit - 1));
% 	
else
	error('reach the end of file');
end

if obj.step == obj.imLength
	obj.eof = true;
end

time = obj.step;