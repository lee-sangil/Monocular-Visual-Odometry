function [time, image_left, image_right] = get_current_image(obj)

if obj.step < obj.imLength
	obj.step = obj.step + 1;
	
	left_filename = [ obj.imLeftDir sprintf( '%.10d.png', obj.step + obj.imInit - 1) ];
	right_filename = [ obj.imRightDir sprintf( '%.10d.png', obj.step + obj.imInit - 1) ];
	image_left = imread(left_filename);
	image_right = imread(right_filename);
	
else 
	error('reach the end of file'); 
end

if obj.step == obj.imLength
	obj.eof = true;
end

time = obj.step;