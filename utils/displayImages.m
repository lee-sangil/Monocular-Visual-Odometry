clc
clear
close all
clearvars -global

cd ..
addpath(genpath('utils'));

profile off
profile on

%% PACKAGE CLASS
pkg = ICSL('F:\Datasets\ICSL-DE\rgbd_dataset_slow_object\');
read(pkg);

% vidObj1 = VideoWriter('image_left', 'MPEG-4');
% vidObj2 = VideoWriter('image_right', 'MPEG-4');
vidObj3 = VideoWriter('depth', 'MPEG-4');

% set(vidObj1, 'FrameRate', 20);
% set(vidObj2, 'FrameRate', 20);
set(vidObj3, 'FrameRate', 20);

% open(vidObj1);
% open(vidObj2);
open(vidObj3);

% figure(1);
% f1 = imshow(zeros(fliplr(pkg.get_im_size)));
% figure(2);
% f2 = imshow(zeros(fliplr(pkg.get_im_size)));
figure(3);
f3 = imshow(zeros(fliplr(pkg.get_im_size)));%colormap(jet(256));caxis([0 5000]);

for i = 1:pkg.get_im_length
	[~, image_left, depth] = get_current_image(pkg);
	depth = im2double(depth) * 0.001 * 65535 / 1.5;
	
% 	set(f1, 'CData', im2double(image_left));
% 	writeVideo(vidObj1, getframe(figure(1)));
	
% 	set(f2, 'CData', im2double(image_right));
% 	writeVideo(vidObj2, getframe(figure(2)));
	
	set(f3, 'CData', im2double(depth));
	writeVideo(vidObj3, getframe(figure(3)));
	
	drawnow;
end

% close(vidObj1);
% close(vidObj2);
close(vidObj3);