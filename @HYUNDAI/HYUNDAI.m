classdef HYUNDAI < handle
	properties (Access = public)
		type
		
		identifier
		BaseDir
		imgDir
		
		imgList
		
		imInit
		imLength
		
		K
		radialDistortion
		tangentialDistortion
		imSize
		
		% iteration
		step
		
		% bool
		eof
	end
	methods (Access = public)
		% CONSTRUCTOR
		function obj = HYUNDAI(varargin)
			
			obj.type = 'mono';
			obj.step = 0;
			obj.eof = false;
			
			if ischar(varargin{1})
				% Default values
				obj.identifier = 'hyundai';
				obj.imInit = 1;
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
		[time, image] = get_current_image(obj)
		eof = end_of_file(obj)
		
		% SET function
		obj = read(obj)
		obj = setupParams(obj, varargin)

		[idx1, idx2] = synchronize(obj, t1, t2)
	end
end