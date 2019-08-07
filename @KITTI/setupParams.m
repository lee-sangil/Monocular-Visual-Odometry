function obj = setupParams(obj, varargin)
% the dataset can be downloaded from:
% http://www.cvlibs.net/datasets/kitti/raw_data.php
% pleas select the [synced data] version

switch( varargin{1} )
	case 1 % KITTI case 1
		BaseDir = 'F:/Datasets/KITTI/2011_10_03/2011_10_03_drive_0027_sync/';
		calibDir = 'F:/Datasets/KITTI/2011_10_03/';
		rectified = true;
		
		imInit      = 0;   % initial image number, 0-base index
		imLength    = 1200; % number of image
		
		
	case 2 % KITTI case 2
		BaseDir = 'F:/Datasets/KITTI/2011_10_03/2011_10_03_drive_0042_sync/';
		calibDir = 'F:/Datasets/KITTI/2011_10_03/';
		rectified = true;
		
		imInit      = 0; % initial image number, 0-base index
		imLength    = 1000; % number of image
		
		
	case 3 % KITTI case 3
		BaseDir = 'F:/Datasets/KITTI/2011_10_03/2011_10_03_drive_0047_sync/';
		calibDir = 'F:/Datasets/KITTI/2011_10_03/';
		rectified = true;
		
		imInit      = 0; % initial image number, 0-base index
		imLength    = 830; % number of image
		
		
	case 4 % KITTI case 4
		BaseDir = 'F:/Datasets/KITTI/2011_09_26/2011_09_26_drive_0052_sync/';
		calibDir = 'F:/Datasets/KITTI/2011_09_26/';
		rectified = true;
		
		imInit      = 0; % initial image number, 0-base index
		imLength    = 75; % number of image
		
	case 5 % KITTI case 4
		BaseDir = 'F:/Datasets/KITTI/2011_09_26/2011_09_26_drive_0017_sync/';
		calibDir = 'F:/Datasets/KITTI/2011_09_26/';
		rectified = true;
		
		imInit      = 0; % initial image number, 0-base index
		imLength    = 110; % number of image
		
	case 6 % KITTI case 4
		BaseDir = 'F:/Datasets/KITTI/2011_09_26/2011_09_26_drive_0018_sync/';
		calibDir = 'F:/Datasets/KITTI/2011_09_26/';
		rectified = true;
		
		imInit      = 0; % initial image number, 0-base index
		imLength    = 260; % number of image
		
	case 7 % KITTI case 4
		BaseDir = '/home/icsl/Documents/dataset/2011_09_26/2011_09_26_drive_0035_sync/';
		calibDir = '/home/icsl/Documents/dataset/2011_09_26/';
		rectified = true;
		
		imInit      = 0; % initial image number, 0-base index
		imLength    = 100; % number of image
		
	case 8
		BaseDir = 'E:\Datasets\KITTI\2011_10_03\2011_10_03_drive_0027_sync\';
		calibDir = 'E:\Datasets\KITTI\2011_10_03\';
		rectified = true;
		
		imInit      = 0; % initial image number, 0-base index
		imLength    = 700; % number of image
		
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
obj.BaseDir = BaseDir;
obj.calibDir = calibDir;
obj.rectified = rectified;
obj.imInit = imInit;
obj.imLength = imLength;
