function obj = read(obj)

BaseDir = obj.BaseDir;
calibDir = obj.calibDir;
imInit = obj.imInit;
imLength = obj.imLength;

imLeftDir = [ BaseDir 'image_00/data/' ];
imRightDir = [ BaseDir 'image_01/data/' ];

cd utils/devkit
camCalib = loadCalibrationCamToCam( [ calibDir 'calib_cam_to_cam.txt' ] );
IMU_to_vel_Calib = loadCalibrationRigid( [ calibDir 'calib_imu_to_velo.txt' ] );
vel_to_cam_Calib = loadCalibrationRigid( [ calibDir 'calib_velo_to_cam.txt' ] );
oxtsTab = loadOxtsliteData( BaseDir, imInit+1:imInit+imLength );
pose = convertOxtsToPose( oxtsTab );
cd ../..

if obj.rectified
	imSize = camCalib.S_rect{1};
	K = camCalib.P_rect{1}(1:3,1:3);
else
	imSize = camCalib.S{1};
	K = camCalib.K{1};
	D = camCalib.D{1};
	obj.radialDistortion = D([1 2 5]);
	obj.tangentialDistortion = D([3 4]);
end

% K = camCalib.P_rect{1}*[ camCalib.R_rect{1} zeros(3,1); zeros(1,3) 1 ];
% K = K(1:3,1:3);

IMU_to_cam_Calib = vel_to_cam_Calib*IMU_to_vel_Calib;
Rci = IMU_to_cam_Calib(1:3,1:3);
pci = IMU_to_cam_Calib(1:3,4);

Ric = Rci';
pic = -Ric * pci;

IntrinsicMatrix1 = camCalib.K{1}';
IntrinsicMatrix1(3,1:2) = IntrinsicMatrix1(3,1:2) + 1;
IntrinsicMatrix2 = camCalib.K{2}';
IntrinsicMatrix2(3,1:2) = IntrinsicMatrix2(3,1:2) + 1;

cameraParams1 = cameraParameters('IntrinsicMatrix',IntrinsicMatrix1,'RadialDistortion',camCalib.D{1}([1 2 5]), 'TangentialDistortion', camCalib.D{1}([3 4]));
cameraParams2 = cameraParameters('IntrinsicMatrix',IntrinsicMatrix2,'RadialDistortion',camCalib.D{2}([1 2 5]), 'TangentialDistortion', camCalib.D{2}([3 4]));
stereoParams = stereoParameters(cameraParams1, cameraParams2, inv(camCalib.R{2}), -camCalib.R{2}.'*camCalib.T{2});
% stereoParams = stereoParameters(cameraParams1, cameraParams2, inv(camCalib.R{2}), -camCalib.R{2}.'*camCalib.T{2}*1000);
% stereoParams = stereoParameters(cameraParams1, cameraParams2, camCalib.R{2}, camCalib.T{2});
% stereoParams = stereoParameters(cameraParams1, cameraParams2, eye(3), [0,-100,0]);

%% load ground truth data
RgiTab = zeros( 3, 3, imLength );
pgiTab = zeros( 3, imLength );
qgiTab = zeros( 4, imLength );

RgcTab = zeros( 3, 3, imLength );
pgcTab = zeros( 3, imLength );

for k = 1:imLength
	RgiTab(:,:,k) = pose{k}(1:3,1:3);
	qgiTab(:,k) = convert_r2q(RgiTab(:,:,k));
	pgiTab(:,k) = pose{k}(1:3,4);
	
	RgcTab(:,:,k) = RgiTab(:,:,k)*Ric;
	pgcTab(:,k) = pgiTab(:,k)+RgiTab(:,:,k)*pic;
	
end

%% load IMU Data
vel_index = 9:11; % FLU frame
am_index = 12:14; % 12:14 body frame, 15:17 FLU frame
wm_index = 18:20; % 18:20 body frame, 21:23 FLU frame

IMUTab = zeros(6,imLength);
for k = 1:imLength
	IMUTab(1:3,k) = oxtsTab{k}(wm_index);
	IMUTab(4:6,k) = oxtsTab{k}(am_index);
end

bias = mean(IMUTab(1:3,1:min(50, imLength)),2);
IMUTab(1:3,:) = bsxfun(@minus, IMUTab(1:3,:), bias);

%% 
obj.imLeftDir = imLeftDir;
obj.imRightDir = imRightDir;
obj.imSize = imSize;
obj.K = K;
obj.stereoParams = stereoParams;

obj.Rci = Rci;
obj.pci = pci;
obj.Ric = Ric;
obj.pic = pic;

obj.RgiTab = RgiTab;
obj.pgiTab = pgiTab;
obj.qgiTab = qgiTab;
obj.RgcTab = RgcTab;
obj.pgcTab = pgcTab;

obj.IMUTab = IMUTab;

fprintf('# load: [%s]\n', BaseDir);
fprintf('# imStep: %d...%d\n', obj.imInit+1, obj.imInit+obj.imLength)
