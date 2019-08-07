function obj = read(obj)

DEBUG = false;

imSize = [640 1280];
K = [500 0 imSize(2)/2;
	0 500 imSize(1)/2;
	0 0 1];
imInit = max(1, obj.imInit);

%% Construct landmark points and pose
n = 2500;
diameter = 600;
X0 = diameter*(-0.5+rand(3,n));
ID = 1:n;

Toc = cell(1,obj.imLength);
Poc = zeros(4,obj.imLength);
Toc{1} = eye(4);
Poc(:,1) = [0;0;0;1];
cameraZ0 = [0;0;1];

features = cell(1, obj.imLength);
points = cell(1, obj.imLength);
points_id = cell(1, obj.imLength);
	
for i = 2:50
	w_ = [0.1*(-0.5+rand) 1 0.1*(-0.5+rand)]';
	w = w_/norm(w_);
	theta = 0.01*(-0.5+rand);
	R = expm(skew(w)*theta);
	t_ = [0.1*rand() 0.1*rand() 5]';
	t = t_ / norm(t_);
	
	Toc{i} = Toc{i-1} * [R t; 0 0 0 1];
end

for i = 51:150
	w_ = [0.05*(-0.5+rand) 1 0.05*(-0.5+rand)]';
	w = w_/norm(w_);
	theta = 0.03*(1+0.1*rand);
	R = expm(skew(w)*theta);
	t_ = [0.1*(-0.5+rand) 0.1*(-0.5+rand) 40]';
	t = t_ / norm(t_);
	
	Toc{i} = Toc{i-1} * [R t; 0 0 0 1];
end

for i = 151:200
	w_ = [0.05*(-0.5+rand) 1 0.05*(-0.5+rand)]';
	w = w_/norm(w_);
	theta = 0.005*(-0.5+rand);
	R = expm(skew(w)*theta);
	t = [0.1*(-0.5+rand) 0.1*(-0.5+rand) (obj.imLength-i)]'/(obj.imLength-151);
	
	Toc{i} = Toc{i-1} * [R t; 0 0 0 1];
end

for i = 201:250
	w_ = -0.5+rand(3,1);
	w = w_/norm(w_);
	theta = 0.001*(-0.5+rand);
	R = expm(skew(w)*theta);
	t = 0.001*(-0.5+rand(3,1));
	
	Toc{i} = Toc{i-1} * [R t; 0 0 0 1];
end

for i = 251:obj.imLength
	w_ = [0.05*(-0.5+rand) 1 0.05*(-0.5+rand)]';
	w = w_/norm(w_);
	theta = 0.03*(1+0.1*rand);
	R = expm(skew(w)*theta);
	t = [0.1*(-0.5+rand) 0.1*(-0.5+rand) (i-231)]'/10;
	
	Toc{i} = Toc{i-1} * [R t; 0 0 0 1];
end

%% Construct projected features and points

if DEBUG
	fig = figure();
	subplot(121);
	scatter3(X0(1,:),X0(2,:),X0(3,:),20, 'filled', 'MarkerFaceColor', [0 0 0], 'MarkerFaceAlpha', .2); hold on;
	hs = scatter3([],[],[],20, 'filled', 'MarkerFaceColor', [1 0 0]);
	hax = draw_camera([], Toc{1}, [], true, diameter/2);
	hp = plot3(Poc(1,1), Poc(2,1), Poc(3,1), 'r-', 'LineWidth', 2);
	xlim([-diameter/2 diameter/2]);
	ylim([-diameter/2 diameter/2]);
	zlim([-diameter/2 diameter/2]);
	
	subplot(122);
	huv = scatter([], [], 'k.');
	xlim([0 imSize(2)]);
	ylim([0 imSize(1)]);
	
	set(gcf, 'Position', [554 472 971 420]);
end

for i = 1:obj.imLength
	Poc(:,i) = Toc{i} * [0;0;0;1];
	cameraZ = Toc{i}(1:3,1:3)*cameraZ0;
	
	X = Toc{i}\[X0;ones(1,length(X0))];
	x = X(1:3,:) ./ X(3,:);
	uv = K*x;
	
	forwardCamera = cameraZ.'*X0 > cameraZ.'*Poc(1:3,i);
	insideImageSensor = uv(1,:) > 1 & uv(1,:) < imSize(2) & uv(2,:) > 1 & uv(2,:) < imSize(1);
	valid = forwardCamera & insideImageSensor;
	features{i} = uv(1:2,valid) + 0.01*(-0.5+rand);
	points{i} = X(1:3,valid);
	points_id{i} = ID(valid);
	
	if DEBUG && ~mod(i,round(obj.imLength/60))
		set(hs, 'XData', X0(1,valid), 'YData', X0(2,valid), 'ZData', X0(3,valid));
		hax = draw_camera(hax, Toc{i});
		set(hp, 'XData', Poc(1,1:i), 'YData', Poc(2,1:i), 'ZData', Poc(3,1:i));
		set(huv, 'XData', uv(1,valid), 'YData', uv(2,valid));
		drawnow;
	end
end

%%
obj.imSize = fliplr(imSize);
obj.imInit = imInit;
obj.features = features;
obj.points = points;
obj.points_id = points_id;
obj.pose = Toc{imInit}\Poc(:,imInit:end);

obj.K = K;

fprintf('# load landmarks\n');

if DEBUG
	close(fig);
end