step = 0;

mkdir('sequences/rgb/');
mkdir('sequences/depth/');

while step < pkg.imLength
	
	step = step + 1;
	
	rgb_filename = pkg.rgbList{step + pkg.imInit - 1};
	depth_filename = pkg.depthList{step + pkg.imInit - 1};
	
	image = imread(rgb_filename);
	depth = imread(depth_filename);
	
	group = vo.group(:,:,step);
	model = vo.model(:,:,step);
% 	mask = isnan(model) | model == 1;
	mask = model == 1 & group == 1;
	mask = kron(mask, ones(16));
	depth(~mask) = 0;
	
	imwrite(image, ['sequences/rgb/' rgb_filename(end-20:end)]);
	imwrite(depth, ['sequences/depth/' depth_filename(end-20:end)]);
	
	disp(step);
end