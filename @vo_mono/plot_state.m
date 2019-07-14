function obj = plot_state(obj, plot_initialized)

persistent h_image h_inlier h_outlier line_inlier line_outlier h_traj h_curr h_point h_point_0 sfig1 sfig2

% Compensate a parameter "step", cause obj.step grows at the last stage of vo.run()
step = obj.step - 1;

% Get uv-pixel point to paint over the image
features = obj.features;

max_len = max(max([obj.features(:).life]), 2);
uv = {obj.features(:).uv};
uv = cellfun(@(x) fill_nan(x, max_len), uv, 'un', 0);

uvArr = zeros(2, length(uv), max_len);
for i = 1:max_len
	uvIdx = cellfun(@(x) column(x, i), uv, 'un', 0);
	uvArr(:,:,i) = cell2mat(uvIdx);
end

% Get initialized 3D points, p_0, and observed 3D point (p_k) of each step in the
% initizlied coordinates, p_k_0
p_k = [features(:).point];
p_k_0 = nan(size(p_k));
p_0 = [obj.features(:).point_init];
p_life = [obj.features(:).life];
p_std = sqrt([obj.features(:).point_prev_var]);

[idx, ~] = seek_index(obj, obj.nFeature, [obj.features(:).is_3D_reconstructed]);
for i = idx
	p_k_0(:,i) = obj.TocRec{step-1} \ p_k(:,i);
end

% Get inlier pixel coordinates
arrIdx = 1:length(features);
[inlierIdx, ~] = seek_index(obj, obj.nFeature3DReconstructed, [obj.features(:).is_3D_reconstructed]);
% [inlierIdx, ~] = seek_index(obj, obj.nFeatureMatched, [obj.features(:).is_matched]);
outlierIdx = arrIdx(~ismember(arrIdx, inlierIdx));

scale = norm(obj.TRec{step}(1:3,4));

%% Initialize figure 1: subs1, subs2
if ~plot_initialized
	
	fig1 = figure(1);
	
	sfig1 = subplot(1,2,1);
	h_image = imshow(uint8(obj.cur_image)); hold on;
	
	for i = 1:obj.bucket.grid(1)-1
		x = floor(i*obj.params.imSize(1)/obj.bucket.grid(1));
		line([x x], [0 obj.params.imSize(2)], 'linestyle', ':', 'color', [.5 .5 .5]);
	end
	
	for j = 1:obj.bucket.grid(2)-1
		y = floor(j*obj.params.imSize(2)/obj.bucket.grid(2));
		line([0 obj.params.imSize(1)], [y y], 'linestyle', ':', 'Color', [.5 .5 .5]);
	end
	
	h_inlier = scatter( uvArr(1,inlierIdx,1), uvArr(2,inlierIdx,1), 'ro' );
	h_outlier = scatter( uvArr(1,outlierIdx,1), uvArr(2,outlierIdx,1), 'bo' );
	
	Xin = zeros(2*(max_len-1), length(inlierIdx));
	Yin = zeros(size(Xin));
	Xout = zeros(2*(max_len-1), length(outlierIdx));
	Yout = zeros(size(Xout));
	for i = 1:max_len-1
		Xin(2*i-1:2*i, :) = [uvArr(1,inlierIdx,i+1); uvArr(1,inlierIdx,i)];
		Yin(2*i-1:2*i, :) = [uvArr(2,inlierIdx,i+1); uvArr(2,inlierIdx,i)];
		Xout(2*i-1:2*i, :) = [uvArr(1,outlierIdx,i+1); uvArr(1,outlierIdx,i)];
		Yout(2*i-1:2*i, :) = [uvArr(2,outlierIdx,i+1); uvArr(2,outlierIdx,i)];
	end
	
	line_inlier = line( sfig1, Xin, Yin, 'Color', 'y');
	line_outlier = line( sfig1, Xout, Yout, 'Color', 'b');
	
	%
	sfig2 = subplot(122);
	h_traj = plot(obj.PocRec(1,1:step), obj.PocRec(3,1:step), 'r-');axis square equal;hold on
	h_curr = scatter(obj.PocRec(1,step), obj.PocRec(3,step), 'ro');
	h_point = scatter3( p_k_0(1,:), p_k_0(2,:), p_k_0(3,:), 'k.');
	h_point_0 = scatter3( p_0(1,:), p_0(2,:), p_0(3,:), 'ko' );
	
	xlim([obj.PocRec(1,step)-25 obj.PocRec(1,step)+25]);
	ylim([obj.PocRec(3,step)-25 obj.PocRec(3,step)+25]);
	grid on;
	
	colormap(sfig2, cool);
	caxis([0 10]);
	
	set(fig1, 'Position', [55 253 1836 460]);
	set(sfig1, 'Position', [0.05 0.1 0.7 0.8]);
	set(sfig2, 'Position', [0.75 0.1 0.2 0.8]);
	set(sfig2, 'XMinorGrid', 'on');
	set(sfig2, 'YMinorGrid', 'on');
	
	w = 0;
else
	%% FIGURE 1: sub1 - image and features, sub2 - xz-trajectory
	set(h_image, 'CData', obj.cur_image);
	
	set(h_inlier, 'XData', uvArr(1,inlierIdx,1), 'YData', uvArr(2,inlierIdx,1));
	set(h_outlier, 'XData', uvArr(1,outlierIdx,1), 'YData', uvArr(2,outlierIdx,1));
	
	delete(line_inlier);
	delete(line_outlier);
	
	Xin = zeros(2*(max_len-1), length(inlierIdx));
	Yin = zeros(size(Xin));
	Xout = zeros(2*(max_len-1), length(outlierIdx));
	Yout = zeros(size(Xout));
	for i = 1:max_len-1
		Xin(2*i-1:2*i, :) = [uvArr(1,inlierIdx,i+1); uvArr(1,inlierIdx,i)];
		Yin(2*i-1:2*i, :) = [uvArr(2,inlierIdx,i+1); uvArr(2,inlierIdx,i)];
		Xout(2*i-1:2*i, :) = [uvArr(1,outlierIdx,i+1); uvArr(1,outlierIdx,i)];
		Yout(2*i-1:2*i, :) = [uvArr(2,outlierIdx,i+1); uvArr(2,outlierIdx,i)];
	end
	
	line_inlier = line( sfig1, Xin, Yin, 'Color', 'y');
	line_outlier = line( sfig1, Xout, Yout, 'Color', 'b');
	
	set(h_traj, 'XData', obj.PocRec(1,1:step), 'YData', obj.PocRec(3,1:step));
	set(h_curr, 'XData', obj.PocRec(1,step), 'YData', obj.PocRec(3,step));
	set(h_point, 'XData', p_k_0(1,inlierIdx), 'YData', p_k_0(3,inlierIdx), 'ZData', -p_k_0(2,inlierIdx));
	set(h_point_0, 'XData', p_0(1,inlierIdx), 'YData', p_0(3,inlierIdx), 'ZData', -p_0(2,inlierIdx),'CData', p_life(inlierIdx), 'LineWidth', 1.5);
		
	xlim(sfig2, [obj.PocRec(1,step)-25*scale obj.PocRec(1,step)+25*scale]);
	ylim(sfig2, [obj.PocRec(3,step)-25*scale obj.PocRec(3,step)+25*scale]);
	
	currentunits = get(sfig2, 'Units');
	set(sfig2, 'Units', 'Points');
	axpos = get(sfig2, 'Position');
	set(sfig2, 'Units', currentunits);
	markerWidth = p_std(inlierIdx) * 5.58; %/ diff(xlim) * axpos(3); % Calculate Marker width in points
	set(h_point_0, 'SizeData', markerWidth.^2);
	
	colormap(sfig2, cool);
	caxis([0 10]);
end

%% FIGURE 2: weight for choosing buckets from feature density
weight = 0.05 ./ (obj.bucket.prob + 0.05);
fig2 = figure(2);
subplot(2,3,[1 2]);
title('Weight for choosing buckets');
surf(weight/sum(weight(:)));
caxis([0 1/sqrt(prod(obj.bucket.grid))]);
axis equal
view([90 90]);
xlim([1 obj.bucket.grid(2)]);
ylim([1 obj.bucket.grid(1)]);

%% FIGURE 3: histogram of norm of motion vectors from klt tracker
subplot(2,3,4);
motion = uvArr(:,1:obj.nFeatureMatched,1) - uvArr(:,1:obj.nFeatureMatched,2);
histogram(sqrt(sum(motion.^2)), 20);
title('Norm of motion vectors');
grid on

% %% FIGURE 4: scale
% subplot(2,3,5);
% if step >= 2
% 	TRec = [obj.TRec{2:step}];
% 	scale = sqrt(sum(TRec(1:3, 4:4:end).^2, 1));
% 	plot(2:step, scale, 'LineWidth', 1.5);
% 	title('Scale propagation');
% 	ylim([0 5]);
% 	grid on
% end

% %% FIGURE 5: histogram of error of features from essential evaluation
% subplot(2,3,6);
% if obj.nFeature2DInliered >= obj.params.thInlier
% 	E = obj.essential;
% 	x1 = (obj.params.K \ [uvArr(:,1:obj.nFeatureMatched,2); ones(1,obj.nFeatureMatched)]);
% 	x2 = (obj.params.K \ [uvArr(:,1:obj.nFeatureMatched,1); ones(1,obj.nFeatureMatched)]);
% 	Ex1 = E * x1;
% 	Etx2 = E.' * x2;
% 	
% 	dist = diag(x2.'*E*x1).^2 ./ ( Ex1(1,:).^2 + Ex1(2,:).^2 + Etx2(1,:).^2 + Etx2(2,:).^2 ).';
% 	histogram(dist, 40);
% 	title('Error of essential constraint');
% 	
% 	grid on
% end

% %% FIGURE 6: 3D graph of error of features from essential evaluation
% subplot(2,3,3);
% if obj.nFeature2DInliered >= obj.params.thInlier
% 	E = obj.essential;
% 	x1 = (obj.params.K \ [uvArr(:,1:obj.nFeatureMatched,2); ones(1,obj.nFeatureMatched)]);
% 	x2 = (obj.params.K \ [uvArr(:,1:obj.nFeatureMatched,1); ones(1,obj.nFeatureMatched)]);
% 	Ex1 = E * x1;
% 	Etx2 = E.' * x2;
% 	
% 	dist = diag(x2.'*E*x1).^2 ./ ( Ex1(1,:).^2 + Ex1(2,:).^2 + Etx2(1,:).^2 + Etx2(2,:).^2 ).';
% 	
% 	uv = round(uvArr(1:2,1:obj.nFeatureMatched,2)).';
% 	[uv, idx] = unique(uv, 'rows');
% 	d = dist(idx);
% 	
% 	F = scatteredInterpolant(uv,d);
% 	F.Method = 'natural';
% 	
% 	u_interp = 1:obj.params.imSize(1);
% 	v_interp = 1:obj.params.imSize(2);
% 	
% 	[U_interp, V_interp] = meshgrid(u_interp, v_interp);
% 	d_interp = F(U_interp, V_interp);
% 	mesh(U_interp, V_interp, d_interp);
% 	title('Error of essential constraint');
% 	view([-360 -90]);
% 	caxis([0 1e-4]);
% 	axis equal tight
% 	grid on
% end
cur_pos = get(fig2, 'Position');
set(fig2, 'Position', [cur_pos(1:2) 960 480]);
drawnow;

end

function x = fill_nan(x, max_len)

len = size(x, 2);
x(:, len+1:max_len) = nan;

end

function x = column(x, idx)

x = x(:,idx);

end

