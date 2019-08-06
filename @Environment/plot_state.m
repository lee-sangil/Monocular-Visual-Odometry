function obj = plot_state( obj, plot_initialized, vo, pkg, params )
%%PLOTOVERVIEW Pretty plotting 
%
% INPUT:
%   - curr_image(H, W): grey scale image matrix at current time t
%   - curr_state(struct): inculdes state.P(2xk) & state.X(3xk)
%   - trajectory(3, N): matrix storing entire estimated trajectory poses
%   - pointcloud(N): cells containing the landmarks detected at each pose
%   - num_of_latest_states: number of recent poses to display
%   - params: pretty plotting parameters
%   - varargin: Arguments to print ground truth
%
% NOTE: 
% trajectory: Index 1 corresponds to the most recent entry 
% pointcloud: Last index corresponds to the most recent entry

persistent trajectory
persistent pointcloud_cell_index pointcloud

if ~plot_initialized
	pointcloud_cell_index = 1;
	trajectory = [];
end
GTExist = isprop(pkg, 'pose');

curr_image = vo.curr_image;
curr_state = vo.state;
num_of_latest_states = params.num_of_latest_states;

% Check camera pose
if (~isempty(vo.pose))
	% append to the trajectory cell
	trajectory = [reshape(vo.pose, [12, 1]), trajectory];
	% Save landmarks from latest 20 states
	% TODO! Fix: Inefficient hack to make plotting easier
	if pointcloud_cell_index > num_of_latest_states
		for j = 1:(num_of_latest_states - 1)
			pointcloud{j} = pointcloud{j+1};
		end
		pointcloud_cell_index = num_of_latest_states;
	end
	pointcloud{pointcloud_cell_index} = curr_state.X;
	pointcloud_cell_index = pointcloud_cell_index + 1;
else
	warning(['Frame ' num2str(vo.step) ' failed tracking!']);
end

% count minima of number of frames to be plotted and current status
num_frames_parsed = min(num_of_latest_states, size(trajectory, 2));
%% Show image with keypoints tracking
sp_1 = subplot(2, 3, [1 2]);
cla(sp_1);
imshow(curr_image);
hold on;      
% convert matrix shape from M x 2 to 2 x M
if ~isempty(curr_state.C)
    candi_keypoints = curr_state.C';
    plot(candi_keypoints(1, :), candi_keypoints(2, :), '.r', 'LineWidth', 2)
end
curr_keypoints = curr_state.P';
plot(curr_keypoints(1, :), curr_keypoints(2, :), '.g', 'LineWidth', 2)
title('Current frame (green: tracked keypoints, red: candidate keypoints)');  

%% Plot latest coordinate and landmarks
sp_2 = subplot(2, 3, [3, 6]); 
cla(sp_2);
hold on;  grid on;  
axis vis3d;
% plotting landmarks
for i = 1:num_frames_parsed
    landmarks = pointcloud{i}';
    % convert landmarks array from M x 3 to 3 x M and plot them
    if i ~= num_frames_parsed
        scatter3(landmarks(1, :), landmarks(2, :), landmarks(3, :), 5, [0.86, 0.86, 0.86]);
    end
end

% plotting most recent point cloud
landmarks = pointcloud{num_frames_parsed}';
% convert landmarks array from M x 3 to 3 x M and plot them
scatter3(landmarks(1, :), landmarks(2, :), landmarks(3, :), 5, 'k');
        
% plotting poses (done separately for a reason, believe me)
for i = 1:num_frames_parsed
    pose = reshape(trajectory(:,i), [3,4]);
    R_W_C = pose(:,1:3);
    t_W_C = pose(:,4);
    plotCoordinateFrame(R_W_C, t_W_C, 1);
end

% set axis limit for cleaner plots
t_W_C = trajectory(10:12, 1);
margin = params.margin_local;
axis([t_W_C(1)-margin, t_W_C(1)+margin, ... 
      t_W_C(2)-margin, t_W_C(2)+margin, ...
      t_W_C(3)-margin, t_W_C(3)+margin]);
title(['Landmarks & Coordinates of latest ', num2str(num_of_latest_states),' frames']);  
set(gcf, 'GraphicsSmoothing', 'on');  
view(0,0);
xlabel('x (in meters)'); ylabel('y (in meters)'); zlabel('z (in meters)');


%% Plot number of tracked landmarks
sp_4 = subplot(2, 3, 4);
cla(sp_4);
count_landmarks = zeros(1, num_frames_parsed);
range = -num_of_latest_states + 1 : 0 ;
for i = 1:num_frames_parsed
    count_landmarks(i) = size(pointcloud{i}, 1);
end
% pad zeros to make range and count landmarks of same length
if num_frames_parsed < num_of_latest_states
    count_landmarks = [zeros(1, num_of_latest_states - num_frames_parsed), count_landmarks];
end
plot(range, count_landmarks);
title(['Status of latest ', num2str(num_of_latest_states),' frames']);  
xlabel('Frame number');
ylabel('Number of landmarks')
ylim([0, params.max_landmarks]);
xlim([-num_of_latest_states + 1, 0])
grid on;

%% Plot full trajectory
sp_3 = subplot(2, 3, 5);
hold on; grid on; 
if size(trajectory, 2) > 1
    plot([trajectory(10, 1), trajectory(10, 2)], [trajectory(12, 1), trajectory(12, 2)],'b.-')
else 
    plot(trajectory(10, 1), trajectory(12, 1),'b.-')
end
title('Full Trajectory');
xlabel('x (in meters)'); ylabel('z (in meters)');
% plot ground truth if argument received
if GTExist
    plot(pkg.pose(:,1), pkg.pose(:,2),'r.-')
    legend('Estimated Pose', 'Ground Truth');
end
% set axis limit for cleaner plots
margin = params.margin_full;
min_x = min(trajectory(10, :));
max_x = max(trajectory(10, :));
min_z = min(trajectory(12, :));
max_z = max(trajectory(12, :));
axis([min_x-margin, max_x+margin, min_z-margin, max_z+margin]);

drawnow;
end