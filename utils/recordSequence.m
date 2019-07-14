clc
clear
close all
clearvars -global

addpath(genpath('utils'));

profile off
profile on

%% PACKAGE CLASS
pkg = ICSL('F:\Datasets\ICSL\icsl-de-place-items-1\');
read(pkg);

vidColor = VideoWriter('color', 'MPEG-4');
set(vidColor, 'FrameRate', 20);
open(vidColor);

vidDepth = VideoWriter('depth', 'MPEG-4');
set(vidDepth, 'FrameRate', 20);
open(vidDepth);

scale = 2;

figure(1);
img = zeros(pkg.get_im_size)';
hfig = imshow(img);
hcb = colorbar;
caxis([0 scale]);
hcb.Label.String = 'depth (m)';
hcb.Label.FontSize = 12;

figure(2);
color = zeros(pkg.get_im_size)';
hfigC = imshow(color);

for i = 1:pkg.imLength
	[~,image, depth] = get_current_image(pkg);
	
	set(hfig, 'CData', im2double(depth) * 1/1000 * 65535 / scale);
	writeVideo(vidDepth, getframe);
	
	set(hfigC, 'CData', im2double(image));
	writeVideo(vidColor, getframe);
	
	drawnow;
end

close(vidDepth);
close(vidColor);