function obj = read(obj)

BaseDir = obj.BaseDir;
imgDir = [ BaseDir '2019_6_3_CAM_0'];
yamlDir = [ BaseDir 'camera.yaml'];

YAML = yamlread(yamlDir);
if ismember('intrinsic', fieldnames(YAML))
	K = YAML.intrinsic;
else
	K = [YAML.Camera.fx 0 YAML.Camera.cx;
		0 YAML.Camera.fy YAML.Camera.cy;
		0 0 1];
end

obj.radialDistortion = [0 0 0];
obj.tangentialDistortion = [0 0];

if ismember('radialDistortion', fieldnames(YAML))
	obj.radialDistortion = YAML.radialDistortion;
else
	if ismember('k1', fieldnames(YAML.Camera))
		obj.radialDistortion(1) = YAML.Camera.k1;
	end
	if ismember('k2', fieldnames(YAML.Camera))
		obj.radialDistortion(2) = YAML.Camera.k2;
	end
	if ismember('k3', fieldnames(YAML.Camera))
		obj.radialDistortion(3) = YAML.Camera.k3;
	end
end
if ismember('tangentialDistortion', fieldnames(YAML))
	obj.tangentialDistortion = YAML.tangentialDistortion;
else
	if ismember('p1', fieldnames(YAML.Camera))
		obj.tangentialDistortion(1) = YAML.Camera.p1;
	end
	if ismember('p2', fieldnames(YAML.Camera))
		obj.tangentialDistortion(2) = YAML.Camera.p2;
	end
end

%% Load rgb and depth data
imgList = openFiles(imgDir);

imSize = size(imread(imgList{1}));
imSize = fliplr(imSize([1 2]));

imInit = max(1, obj.imInit);
imLength = min(obj.imInit+obj.imLength-1, length(imgList))-obj.imInit+1;

%% 
obj.imgDir = imgDir;
obj.imSize = imSize;
obj.imInit = imInit;
obj.imLength = imLength;
obj.imgList = imgList;

obj.K = K;

fprintf('# load: [%s]\n', BaseDir);
fprintf('# imStep: %d...%d\n', obj.imInit, obj.imInit+obj.imLength-1)