function obj = plot_state(obj, plot_initialized, vo, pkg, param)

persistent sfig1 sfig2 h_image h_inlier h_outlier line_inlier line_outlier h_traj h_curr h_point h_point_0 p_0 p_0_id
persistent h_gt

if ~plot_initialized
	p_0 = [];
	p_0_id = [];
end

GTExist = isprop(pkg, 'pose');

% Compensate a parameter "step", cause obj.step grows at the last stage of vo.run()
step = vo.step - 1;

% Get uv-pixel point to paint over the image
features = vo.features;

x_window = [-200 200] * vo.params.initScale * param.plotScale;
y_window = [-180 220] * vo.params.initScale * param.plotScale;
z_window = [-200 200] * vo.params.initScale * param.plotScale;

max_len = max(max([vo.features(:).life]), 2);
uv = {vo.features(:).uv};
uv = cellfun(@(x) fill_nan(x, max_len), uv, 'un', 0);

uvArr = zeros(2, length(uv), max_len);
for i = 1:max_len
	uvIdx = cellfun(@(x) x(:,i), uv, 'un', 0);
	uvArr(:,:,i) = cell2mat(uvIdx);
end

% Get inlier pixel coordinates
arrIdx = 1:length(features);
inlierIdx = find([vo.features(:).is_3D_reconstructed] == true);
outlierIdx = arrIdx(~ismember(arrIdx, inlierIdx));

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

% Get initialized 3D points, p_0, and observed 3D point (p_k) of each step in the
% initizlied coordinates, p_k_0
p_k = [features(:).point];
p_k_0 = nan(size(p_k));
p0 = [vo.features(:).point_init];
p_id = [vo.features(:).id];
p_0 = [p0(:,~isnan(p0(1,:))) p_0];
p_0_id = [p_id(:,~isnan(p0(1,:))) p_0_id];
[~,unique_idx] = unique(p_0_id);
p_0 = p_0(:,unique_idx);
p_0_id = p_0_id(unique_idx);

p_life = [vo.features(:).life];

idx = find([vo.features(:).is_3D_reconstructed] == true);
for i = idx
	p_k_0(:,i) = vo.TocRec{step-1} * p_k(:,i);
end

% Transform camera coordinate to world coordinate
worldToCam = [1 0 0 0; 0 0 1 0; 0 -1 0 0; 0 0 0 1];

p_0_w = worldToCam * p_0;
p_k_0_w = worldToCam * p_k_0;
Pwc = worldToCam * vo.PocRec;
Twc = worldToCam * vo.TocRec{step};

if GTExist
	Pwc_true = worldToCam * pkg.pose;
end

%% Initialize figure 1: subs1, subs2
if ~plot_initialized
	
	figure(1);
	
	sfig1 = subplot(1,2,1);
	h_image = imshow(uint8(vo.cur_image)); hold on;
	
	for i = 1:vo.bucket.grid(1)-1
		x = floor(i*vo.params.imSize(1)/vo.bucket.grid(1));
		line([x x], [0 vo.params.imSize(2)], 'linestyle', ':', 'color', [.5 .5 .5]);
	end
	
	for j = 1:vo.bucket.grid(2)-1
		y = floor(j*vo.params.imSize(2)/vo.bucket.grid(2));
		line([0 vo.params.imSize(1)], [y y], 'linestyle', ':', 'Color', [.5 .5 .5]);
	end
	
	h_inlier = scatter( uvArr(1,inlierIdx,1), uvArr(2,inlierIdx,1), 'ro' );
	h_outlier = scatter( uvArr(1,outlierIdx,1), uvArr(2,outlierIdx,1), 'bo' );
	
	line_inlier = line( Xin, Yin, 'Color', 'y');
	line_outlier = line( Xout, Yout, 'Color', 'b');
	
	%
	sfig2 = subplot(122);
	h_point_0 = scatter3(sfig2, p_0_w(1,:), p_0_w(2,:), p_0_w(3,:), 5, 'filled', 'MarkerFaceColor', [0 0 0], 'MarkerFaceAlpha', .5);hold on;
	h_point = scatter3( p_k_0_w(1,:), p_k_0_w(2,:), p_k_0_w(3,:), 7, 'filled');
	h_traj = plot3(Pwc(1,1:step), Pwc(2,1:step), Pwc(3,1:step), 'k-', 'LineWidth', 2);hold on;
	h_curr = draw_camera([], Twc, 'k', true, 150 * vo.params.initScale * param.plotScale);
	
	if GTExist
		h_gt = plot3(Pwc_true(1,1:step), Pwc_true(2,1:step), Pwc_true(3,1:step), 'r-', 'LineWidth', 2);hold on;
	end
	
	axis square equal;grid on;
	
	xlim(sfig2, Pwc(1,step)+x_window);
	ylim(sfig2, Pwc(2,step)+y_window);
	zlim(sfig2, z_window);
% 	xlim(sfig2, x_window);
% 	ylim(sfig2, y_window);
% 	set(sfig2, 'View', [0 90]);
	set(sfig2, 'View', [-65 15]);
	colormap(sfig2, cool);
	caxis([0 10]);
	
	set(gcf, 'Position', [7 510 1280 480]);
	set(sfig1, 'Position', [0.05 0.05 0.5 0.9]);
	set(sfig2, 'Position', [0.64 0.11 0.3 0.8]);
	set(sfig2, 'XMinorGrid', 'on');
	set(sfig2, 'YMinorGrid', 'on');
	
else
	% FIGURE 1: sub1 - image and features, sub2 - xz-trajectory
	set(h_image, 'CData', vo.cur_image);
	
	set(h_inlier, 'XData', uvArr(1,inlierIdx,1), 'YData', uvArr(2,inlierIdx,1));
	set(h_outlier, 'XData', uvArr(1,outlierIdx,1), 'YData', uvArr(2,outlierIdx,1));
	
	delete(line_inlier);
	delete(line_outlier);
	
	line_inlier = line( sfig1, Xin, Yin, 'Color', 'y');
	line_outlier = line( sfig1, Xout, Yout, 'Color', 'b');
	
	set(h_point_0, 'XData', p_0_w(1,:), 'YData', p_0_w(2,:), 'ZData', p_0_w(3,:));
	set(h_point, 'XData', p_k_0_w(1,inlierIdx), 'YData', p_k_0_w(2,inlierIdx), 'ZData', p_k_0_w(3,inlierIdx),'CData', p_life(inlierIdx));
	set(h_traj, 'XData', Pwc(1,1:step), 'YData', Pwc(2,1:step), 'ZData', Pwc(3,1:step));
	h_curr = draw_camera(h_curr, Twc);
	
	if GTExist
		set(h_gt, 'XData', Pwc_true(1,1:step), 'YData', Pwc_true(2,1:step), 'ZData', Pwc_true(3,1:step));
	end
	
	xlim(sfig2, Pwc(1,step)+x_window);
	ylim(sfig2, Pwc(2,step)+y_window);
	
end

% %%
% % FIGURE 2: weight for choosing buckets from feature density
% weight = 0.05 ./ (obj.bucket.prob + 0.05);
% figure(2);
% subplot(2,3,[1 2]);
% surf(weight/sum(weight(:)));
% title('Weight for choosing buckets');
% caxis([0 1/sqrt(prod(obj.bucket.grid))]);
% view([90 90]);
% xlim([1 obj.bucket.grid(2)]);
% ylim([1 obj.bucket.grid(1)]);
% 
% % FIGURE 3: histogram of norm of motion vectors from klt tracker
% subplot(2,3,4);
% motion = uvArr(:,1:obj.nFeatureMatched,1) - uvArr(:,1:obj.nFeatureMatched,2);
% histogram(sqrt(sum(motion.^2)), 20);
% title('Norm of motion vectors');
% grid on
% 
% % FIGURE 4: scale
% subplot(2,3,5);
% if step >= 2
% 	TRec = [obj.TRec{2:step}];
% 	scale = sqrt(sum(TRec(1:3, 4:4:end).^2, 1));
% 	plot(2:step, scale, 'LineWidth', 1.5);
% 	title('Scale propagation');
% 	ylim([0 5]);
% 	grid on
% end
% 
% % FIGURE 5: histogram of error of features from essential evaluation
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
% 
% % FIGURE 6: 3D graph of error of features from essential evaluation
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
% 	axis tight
% 	grid on
% end
% 
% set(gcf, 'Position', [76 77 1107 420]);

%%
drawnow;

end

function x = fill_nan(x, max_len)

len = size(x, 2);
x(:, len+1:max_len) = nan;

end

function x = column(x, idx)

x = x(:,idx);

end

