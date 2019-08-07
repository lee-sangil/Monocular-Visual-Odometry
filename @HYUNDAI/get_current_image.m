function [time, image] = get_current_image(obj)

if obj.step < obj.imLength
	obj.step = obj.step + 1;
	
	imgFilename = obj.imgList{obj.step + obj.imInit - 1};
	image = imread(imgFilename);
	
	time = 0;
	
else 
	error('reach the end of file'); 
end 

if obj.step == obj.imLength
	obj.eof = true;
end

end