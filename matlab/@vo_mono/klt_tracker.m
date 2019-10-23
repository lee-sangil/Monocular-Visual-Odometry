function [new_pts, validity, scores] = klt_tracker( obj )

pts = {obj.features(:).uv};
pts = cellfun(@(x) x(:,1), pts, 'un', 0);

bwd_pts_ = [pts{:}];
pts = cellfun(@transpose, pts, 'un', 0);

% Forward-backward error evaluation
fwd_pts = cv.calcOpticalFlowPyrLK(obj.prev_image, obj.cur_image, pts);
bwd_pts = cv.calcOpticalFlowPyrLK(obj.cur_image, obj.prev_image, fwd_pts);

% Convert {[u1,v1], [u2,v2], ...} => {[u1;v1], [u2;v2], ...}
fwd_pts = cellfun(@transpose, fwd_pts, 'un', 0);
bwd_pts = cellfun(@transpose, bwd_pts, 'un', 0);

% Convert {[u1;v1], [u2;v2], ...} => [u1 u2 ...; v1 v2 ...]
fwd_pts = [fwd_pts{:}];
bwd_pts = [bwd_pts{:}];

% Calculate bi-directional error( = validity )
border_invalid = fwd_pts(1,:) < 0 | fwd_pts(1,:) > obj.params.imSize(1) | ...
				fwd_pts(2,:) < 0 | fwd_pts(2,:) > obj.params.imSize(2);
error_valid = sqrt(sum((bwd_pts_ - bwd_pts).^2, 1)) < min( sqrt(sum((bwd_pts_ - fwd_pts).^2, 1))/5, 1);

% Calculate score for each point


% Assign output variable
new_pts = fwd_pts;
validity = ~border_invalid & error_valid;