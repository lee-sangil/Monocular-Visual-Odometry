function obj = set_image( obj, image )

obj.prev_image = obj.cur_image;
obj.cur_image = image;