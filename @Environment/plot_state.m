function obj = plot_state(obj, plot_initialized, vo, pkg, param)

persistent sfig1 sfig2 fig2 h_image h_inlier h_inlierpoint h_outlier line_inlier line_outlier h_traj h_curr h_point h_point_0 p_0 p_0_id
persistent h_gt h_text h_map
persistent h_vel h_velpoint
persistent worldToCam meterToDegree

if ~plot_initialized
	p_0 = [];
	p_0_id = [];
	
	% Transform camera coordinate to world coordinate
	inertialToCam = [1 0 0 0; 0 0 1 0; 0 -1 0 0; 0 0 0 1];
	worldToInertial = [rotz(-52*pi/180) [126.952277;37.531870;0];0 0 0 1];
	worldToCam = worldToInertial * inertialToCam;
	meterToDegree = 1/111111 * diag([1/cos(37.531832 * pi/180) 1 1]); % x-longitude, y-latitude
end

GTExist = isprop(pkg, 'pose');
step = vo.step;

% Get uv-pixel point to paint over the image
features = vo.features;

x_window = [-190 190] * vo.params.initScale * param.plotScale;
y_window = [-180 220] * vo.params.initScale * param.plotScale;
z_window = [-200 200] * vo.params.initScale * param.plotScale;

uvArr = zeros(2,vo.nFeature);
for i = 1:vo.nFeature
	uvArr(:,i) = vo.features(i).uv(:,1);
end

% Get inlier pixel coordinates
arrIdx = 1:length(features);
inlierIdx = find([vo.features(:).is_2D_inliered] == true);
outlierIdx = arrIdx(~ismember(arrIdx, inlierIdx));

% Get initialized 3D points, p_0, and observed 3D point (p_k) of each step in the
% initizlied coordinates, p_k_0
p_k = [features(:).point];
p_k_0 = nan(size(p_k));
p0 = [vo.features(:).point_init];
p0(1:3,:) = meterToDegree * p0(1:3,:);
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

p_0_w = worldToCam * p_0;
p_k_0(1:3,:) = meterToDegree * p_k_0(1:3,:);
p_k_0_w = worldToCam * p_k_0;
PocRec = vo.PocRec;
PocRec(1:3,:) = meterToDegree * PocRec(1:3,:);
Pwc = worldToCam * PocRec;
Twc = vo.TocRec{step};
Twc(1:3,4) = meterToDegree * Twc(1:3,4);
Twc = worldToCam * Twc;

if GTExist
	Pwc_true = worldToCam * meterToDegree * pkg.pose;
end

%% Initialize figure 1: subs1, subs2
if ~plot_initialized
	
	figure(1);
	
	sfig1 = subplot(1,2,1);
	h_image = imagesc(uint8(vo.cur_image)); colormap gray; axis off; hold on;
	
	for i = 1:vo.bucket.grid(1)-1
		x = floor(i*vo.params.imSize(1)/vo.bucket.grid(1));
		line([x x], [0 vo.params.imSize(2)], 'linestyle', ':', 'color', [.5 .5 .5]);
	end
	
	for j = 1:vo.bucket.grid(2)-1
		y = floor(j*vo.params.imSize(2)/vo.bucket.grid(2));
		line([0 vo.params.imSize(1)], [y y], 'linestyle', ':', 'Color', [.5 .5 .5]);
	end
	
	h_inlier = scatter( uvArr(1,inlierIdx), uvArr(2,inlierIdx), 50, 'gs' );
	h_inlierpoint = scatter( uvArr(1,inlierIdx), uvArr(2,inlierIdx), 20, 'g.' );
% 	h_outlier = scatter( uvArr(1,outlierIdx,1), uvArr(2,outlierIdx,1), 50, 'bs' );
	
% 	line_inlier = line( Xin, Yin, 'Color', 'y');
% 	line_outlier = line( Xout, Yout, 'Color', 'b');
	
	%
	sfig2 = subplot(122);
	h_map = plot_google_map('Alpha', 0.5, 'Refresh', 1, 'MapType', 'roadmap', 'APIKey', 'APIKey');axis off;hold on;
	h_point_0 = scatter3(sfig2, p_0_w(1,:), p_0_w(2,:), p_0_w(3,:), 5, 'filled', 'MarkerFaceColor', [0 0 0], 'MarkerFaceAlpha', .5);
	h_point = scatter3( p_k_0_w(1,:), p_k_0_w(2,:), p_k_0_w(3,:), 7, 'filled');
	h_traj = plot3(Pwc(1,1:step), Pwc(2,1:step), Pwc(3,1:step), 'k-', 'LineWidth', 2);hold on;
	h_curr = draw_camera([], Twc, 'k', true, 100 * vo.params.initScale * param.plotScale);
	
	if GTExist
		h_gt = plot3(Pwc_true(1,1:step), Pwc_true(2,1:step), Pwc_true(3,1:step), 'r-', 'LineWidth', 2);hold on;
	end
	
	set(gca, 'XTickLabel', '');
	axis square equal;grid on;
	
	xlim(sfig2, Pwc(1,step)+x_window);
	ylim(sfig2, Pwc(2,step)+y_window);
	zlim(sfig2, z_window);
% 	xlim(sfig2, x_window);
% 	ylim(sfig2, y_window);

	h_text = text(0, 0, 100, vo.status, 'Color', 'red');

	set(sfig2, 'View', [0 90]);
% 	set(sfig2, 'View', [-65 15]);
	colormap(sfig2, cool);
	caxis([0 10]);
	
	set(gcf, 'Position', [7 510 1380 480]);
	set(sfig1, 'Position', [0.02 0.05 0.6 0.9]);
	set(sfig2, 'Position', [0.6 0.05 0.4 0.9]);
	set(sfig2, 'XMinorGrid', 'on');
	set(sfig2, 'YMinorGrid', 'on');
	
	figure(2);
	h_vel = plot(0,0,'k','LineWidth', 2);hold on;
	h_velpoint = plot(step, norm(vo.TRec{step}(1:3,4)), 'ko', 'LineWidth', 2);grid on;
	ylim([0 4]);
	fig2 = gca;
	set(gcf, 'Position', [1394 610 520 380]);
	
	pause(2);
else
	% FIGURE 1: sub1 - image and features, sub2 - xz-trajectory
	set(h_image, 'CData', vo.cur_image);
	
	set(h_inlier, 'XData', uvArr(1,inlierIdx), 'YData', uvArr(2,inlierIdx));
	set(h_inlierpoint, 'XData', uvArr(1,inlierIdx), 'YData', uvArr(2,inlierIdx));
% 	set(h_outlier, 'XData', uvArr(1,outlierIdx,1), 'YData', uvArr(2,outlierIdx,1));
	
% 	delete(line_inlier);
% 	delete(line_outlier);
% 	
% 	line_inlier = line( sfig1, Xin, Yin, 'Color', 'y');
% 	line_outlier = line( sfig1, Xout, Yout, 'Color', 'b');
	
	set(h_point_0, 'XData', p_0_w(1,:), 'YData', p_0_w(2,:), 'ZData', p_0_w(3,:));
	set(h_point, 'XData', p_k_0_w(1,inlierIdx), 'YData', p_k_0_w(2,inlierIdx), 'ZData', p_k_0_w(3,inlierIdx),'CData', p_life(inlierIdx));
	set(h_traj, 'XData', Pwc(1,1:step), 'YData', Pwc(2,1:step), 'ZData', Pwc(3,1:step));
	h_curr = draw_camera(h_curr, Twc);
	
	if GTExist
		set(h_gt, 'XData', Pwc_true(1,1:step), 'YData', Pwc_true(2,1:step), 'ZData', Pwc_true(3,1:step));
	end
	
	xlim(sfig2, Pwc(1,step)+x_window);
	ylim(sfig2, Pwc(2,step)+y_window);
	
	refresh(figure(1));
	set(h_text, 'Position', [Pwc(1,step)+x_window(1)+4, Pwc(2,step)+y_window(1)+5, 100], 'String', ['\bf' vo.status]);
	
	set(h_vel, 'XData', 1:step, 'YData', [get(h_vel, 'YData') norm(vo.TRec{step}(1:3,4))]);
	set(h_velpoint, 'XData', step, 'YData', norm(vo.TRec{step}(1:3,4)));
	xlim(fig2, [max(1, step-9) max(10, step)]);
% 	ylim_ = get(fig2, 'YLim');
% 	if ylim_(2) < norm(vo.TRec{step}(1:3,4))
% 		ylim_(2) = 2*ylim_(2);
% 	end
% 	ylim(fig2, ylim_);
end

drawnow;

end

