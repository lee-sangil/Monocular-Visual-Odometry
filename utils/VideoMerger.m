function VideoMerger(varargin)

[filename, roi, loc] = parse_inputs(varargin{:});
background_color = 255;

%% Get video handles
nPair = nargin / 3;

vidIn = cell(nPair,1);
for i = 1:nPair
	vidIn{i} = VideoReader(filename{i});
end

%% Seg output video handle
vidOut = VideoWriter('out.avi', 'Motion JPEG AVI');
set(vidOut, 'FrameRate', 20);

%% Get image size
x = zeros(nPair,1);
y = zeros(nPair,1);
w = zeros(nPair,1);
h = zeros(nPair,1);
for i = 1:nPair
	x(i) = loc{i}(1);
	y(i) = loc{i}(2);
	w(i) = loc{i}(3);
	h(i) = loc{i}(4);
end

width = max(x+w);
height = max(y+h);

%% Read, warp videos, and write output video
open(vidOut);
isConfirmed = false;
isContinue = true;

step = 1;
while isContinue
	
	image = background_color*ones(height, width, 3, 'uint8');
	for i = 1:nPair
	
		vidFrame = readFrame(vidIn{i});
		roiFrame = imcrop(vidFrame, roi{i});
		warpFrame = imresize(roiFrame, [h(i)+1, w(i)+1]);
		image(y(i):y(i)+h(i), x(i):x(i)+w(i),:) = warpFrame;
		
	end
	
	if ~isConfirmed
		
		fig = imshow(image);
		set(gcf, 'Position', [2100 333 width height]);
		set(gca, 'Position', [0 0 1 1]);
	
		reply = input('Continue? [Y/n]', 's');
		if reply == 'y' || reply == 'Y'
			isConfirmed = true;
		else
			close(vidOut);
			return;
		end
	else
		
		set(fig, 'CData', image);
	end
	
	writeVideo(vidOut, getframe);
	
	for i = 1:nPair
		if ~hasFrame(vidIn{i})
			isContinue = false;
		end
	end
	
	disp(step);
	step = step + 1;
end

close(vidOut);

end

function [filename, roi, loc] = parse_inputs(varargin)

% nargin = length(varargin);
if nargin == 3 && iscell(varargin{1}) && iscell(varargin{2}) && iscell(varargin{3})
	filename = varargin{1};
	roi = varargin{2};
	loc = varargin{3};
else
	if mod(nargin,3) == 0
		nPair = nargin / 3;
		
		filename = cell(nPair, 1);
		roi = cell(nPair, 1);
		loc = cell(nPair, 1);
		
		for i = 1:nPair
			if ~ischar(varargin{i}) || ~isnumeric(varargin{nPair+i}) || ~isnumeric(varargin{2*nPair+i})
				error('Invalid type of inputs.');
			end
			
			filename{i} = varargin{i};
			roi{i} = varargin{nPair+i};
			loc{i} = varargin{2*nPair+i};
		end
	else
		error('Invalid number of inputs.');
	end
end

end