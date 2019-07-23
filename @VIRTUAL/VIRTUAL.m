classdef VIRTUAL < handle
	properties (Access = public)
		type
		identifier
		
		imInit
		imLength
		
		features
		points
		points_id
		pose
		
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
		function obj = VIRTUAL(varargin)
			
			obj.type = 'mono';
			obj.step = 0;
			obj.eof = false;
			
			obj.identifier = 'virtual';
			obj.imInit = 1;
			obj.imLength = 300;
			
			obj.radialDistortion = [0 0 0];
			obj.tangentialDistortion = [0 0];
		end
				
		% GET functions		
		[feature, point, id] = get_current_feature(obj)
		eof = end_of_file(obj)
		
		% SET function
		obj = read(obj)
	end
end