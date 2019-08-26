classdef KITTI < handle
	properties (GetAccess = public, SetAccess = private)
		identifier
		type
		rectified
		
		BaseDir
		calibDir
		imLeftDir
		imRightDir
		
		imInit
		imLength
		
		imSize
		K
		radialDistortion
		tangentialDistortion
		stereoParams
		
		Rci
		pci
		Ric
		pic
		
		RgiTab
		pgiTab
		qgiTab
		RgcTab
		pgcTab
		IMUTab

		% iteration
		step
		
		% bool
		eof
	end
	methods (Access = public)
		% CONSTRUCTOR
		function obj = KITTI(varargin)
			
			obj.type = 'stereo';
			obj.step = 0;
			obj.eof = false;
			
			obj.radialDistortion = [0 0 0];
			obj.tangentialDistortion = [0 0];
			
			if ischar(varargin{1})
				% Default values
				obj.identifier = 'untitled';
				obj.imInit = 0;
				obj.imLength = inf;
				
				for i = 1:length(varargin)
					switch i
						case 1
							obj.BaseDir = varargin{1};
						case 2
							if ~isempty(varargin{2})
								obj.imInit = varargin{2};
							end
						case 3
							if ~isempty(varargin{3})
								obj.imLength = varargin{3};
							end
					end
				end
				
			else
				setupParams(obj, varargin{:});
			end
		end
		
		% GET functions
		BaseDir = get_base_dir(obj)
		calibDir = get_calib_dir(obj)
		imLeftDir = get_im_left_dir(obj)
		imRightDir = get_im_right_dir(obj)
		
		imInit = get_im_init(obj)
		imLength = get_im_length(obj)
		
		imSize = get_im_size(obj)
		K = get_intrinsic(obj)
		identifier = get_identifier(obj)
		
		Rci = get_R_ci(obj)
		pci = get_p_ci(obj)
		Ric = get_R_ic(obj)
		pic = get_p_ic(obj)
		
		RgiTab = get_R_gi_Tab(obj)
		pgiTab = get_p_gi_Tab(obj)
		qgiTab = get_q_gi_Tab(obj)
		RgcTab = get_R_gc_Tab(obj)
		pgcTab = get_p_gc_Tab(obj)
		IMUTab = get_imu_Tab(obj)
		
% 		[image_left, image_right] = get_current_image(obj)
		[time, image_left, image_right] = get_current_image(obj)
		step = get_step(obj)
		eof = end_of_file(obj)
		
		% SET function
		obj = read(obj)
		obj = setupParams(obj, varargin)

	end
end