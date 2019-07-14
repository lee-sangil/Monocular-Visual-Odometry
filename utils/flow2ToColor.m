function img = flow2ToColor(flow, varargin)

UNKNOWN_FLOW_THRESH = 1e3;

u = flow(:,:,1);
v = flow(:,:,2);

maxrad = 50;

% fix unknown flow
idxUnknown = (abs(u)> UNKNOWN_FLOW_THRESH) | (abs(v)> UNKNOWN_FLOW_THRESH);
idxNan = isnan(u) | isnan(v);

u(idxUnknown) = 0;
v(idxUnknown) = 0;

u = u/(maxrad);
v = v/(maxrad);

% compute color

img = computeColor(u, v);  
    
% unknown flow
IDX = repmat(idxUnknown, [1 1 3]);
img(IDX) = 0;

IDX = repmat(idxNan, [1 1 3]);
img(IDX) = 0.94*255;

end