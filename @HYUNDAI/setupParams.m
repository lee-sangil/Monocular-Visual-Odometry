function obj = setupParams(obj, varargin)

switch( varargin{1} )
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%% Vicon Room EuRoC MAV dataset  %%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	case 1
		identifier = 'translation';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil1_translation/';
		
		imInit		= 18;
		imLength	= 100;
		
	case 2
		identifier = 'rotation';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil1_rotation/';
		
		imInit		= 20;
		imLength	= 290;
		
	case 3
		identifier = 'come-and-go';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil1_come-and-go/';
		
		imInit		= 18;
		imLength	= 125;
		
	case 4
		identifier = 'tilt';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil1_tilting/';
		
		imInit		= 1;
		imLength	= 205;
		
	case 5
		identifier = 'translation';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil2_translation/';
		
		imInit		= 40;
		imLength	= 180;
		
	case 6
		identifier = 'rotation';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil2_rotation/';
		
		imInit		= 25;
		imLength	= 300;
		
	case 7
		identifier = 'depth';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil2_depth-change/';
		
		imInit		= 25;
		imLength	= 220;
		
	case 8
		identifier = 'flight1';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil4_flight1/';
		
		imInit		= 60;
		imLength	= 1200;
		
	case 9
		identifier = 'flight2';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil4_flight2/';
		
		imInit		= 320;
		imLength	= 600;
		
	case 10
		identifier = 'obstacle1';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil4_obstacle1/';
		
		imInit		= 175;
		imLength	= 281;
		
	case 11
		identifier = 'obstacle2';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil4_obstacle2/';
		
		imInit		= 40;
		imLength	= 600;
		
	case 12
		identifier = 'flight_with_obstacle';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil4_flight_with_obstacle/';
		
		imInit		= 150;
		imLength	= 350;
		
	case 13
		identifier = 'multiple1';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil5_multiple_object1/';
		
		imInit		= 60;
		imLength	= 400;
		
	case 14
		identifier = 'multiple2';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil5_multiple_object2/';
		
		imInit		= 70;
		imLength	= 350;
		
	case 15
		identifier = 'multiple3';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil5_multiple_object3/';
		
		imInit		= 100;
		imLength	= 440;
		
	case 16
		identifier = '3dObject1';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil5_multiple_object3/';
		
		imInit		= 630;
		imLength	= 240;
		
	case 17
		identifier = '3dObject2';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil5_3d_object/';
		
		imInit		= 208;
		imLength	= 250;
		
	case 18
		identifier = 'multiple4';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil6_multiple_object1/';
		
		imInit		= 45;
		imLength	= 270;
		
	case 19
		identifier = 'multiple5';
		BaseDir = 'F:/Datasets/ICSL/rgbd_dataset_sangil6_multiple_object2/';
		
		imInit		= 40;
		imLength	= 250;
		
	case 20
		identifier = 'vicon1';
		BaseDir = 'F:/Datasets/ICSL/DATA_2017_5_21_15_40_19/';
		
		imInit		= 181;
		imLength	= 1000;
		
	case 21
		identifier = 'vicon2';
		BaseDir = 'F:/Datasets/ICSL/DATA_2017_5_21_15_49_37/';
		
		imInit		= 51;
		imLength	= 840;
		
	case 22
		identifier = 'vicon3';
		BaseDir = 'F:/Datasets/ICSL/DATA_2017_5_21_15_56_16/';
		
		imInit		= 101;
		imLength	= 1250;
		
	case 23
		identifier = 'vicon4';
		BaseDir = 'F:/Datasets/ICSL/DATA_2017_5_22_20_9_50/';
		
		imInit		= 231;
		imLength	= 650;
		
	case 24
		identifier = 'vicon5';
		BaseDir = 'F:/Datasets/ICSL/DATA_2017_5_22_20_20_8/';
		
		imInit		= 483;
		imLength	= 230;
		
	case 25
		identifier = 'vicon6';
		BaseDir = 'F:/Datasets/ICSL/DATA_2017_5_22_20_22_52/';
		
		imInit		= 403;
		imLength	= 1000;
		
	case 26
		identifier = 'odometry_3';
		BaseDir = 'F:/Datasets/ICSL/odometry_3/';
		
		imInit		= 81;
		imLength	= 1500;
		
	case 27
		identifier = 'odometry_4';
		BaseDir = 'F:/Datasets/ICSL/odometry_4/';
		
		imInit		= 31;
		imLength	= 1100;
		
	case 28
		identifier = 'odometry_5';
		BaseDir = 'F:/Datasets/ICSL/odometry_5/';
		
		imInit		= 31;
		imLength	= 1000;
		
	case 29
		identifier = 'odometry_6';
		BaseDir = 'F:/Datasets/ICSL/odometry_6/';
		
		imInit		= 156;
		imLength	= 790;
		
	case 30
		identifier = 'odometry_7';
		BaseDir = 'F:/Datasets/ICSL/odometry_7/';
		
		imInit		= 375;
		imLength	= 1500;
		
	case 31
		identifier = 'odometry_8';
		BaseDir = 'F:/Datasets/ICSL/odometry_8/';
		
		imInit		= 865;
		imLength	= 1840;
		
	case 32
		identifier = 'odometry_9';
		BaseDir = 'F:/Datasets/ICSL/odometry_9/';
		
		imInit		= 641;
		imLength	= 700;
		
	case 33
		identifier = 'odometry_1';
		BaseDir = 'F:/Datasets/ICSL/odometry_1/';
		
		imInit		= 57;
		imLength	= 1100;
		
	case 34
		identifier = 'odometry_2';
		BaseDir = 'F:/Datasets/ICSL/odometry_2/';
		
		imInit		= 81;
		imLength	= 800;
		
	case 35
		identifier = 'odometry_10';
		BaseDir = 'F:/Datasets/ICSL/odometry_10/';
		
		imInit		= 88;
		imLength	= 1100;
		
	case 36
		identifier = 'odometry_11';
		BaseDir = 'F:/Datasets/ICSL/odometry_11/';
		
		imInit		= 240;
		imLength	= 1100;
		
	case 37
		identifier = 'odometry_12';
		BaseDir = 'F:/Datasets/ICSL/odometry_12/';
		
		imInit		= 168;
		imLength	= 1100;
		
	case 38
		identifier = 'odometry_13';
		BaseDir = 'F:/Datasets/ICSL/odometry_13/';
		
		imInit		= 113;
		imLength	= 1100;
		
	otherwise
		error('Unvalid dataset.');

end

% Change imInit and imLength
for i = 2:length(varargin)
	switch i
		case 2
			if ~isempty(varargin{2})
				imInit = varargin{2};
			end
		case 3
			if ~isempty(varargin{3})
				imLength = varargin{3};
			end
	end
end

%%
obj.identifier = identifier;
obj.BaseDir = BaseDir;
obj.imInit = imInit;
obj.imLength = imLength;