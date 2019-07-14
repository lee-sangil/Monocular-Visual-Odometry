function [Iwarp, xImg, yImg, xp, yp, zp] = transform_kerl(IRef, DRef, I, xi, K, bkSize, bk_i, bk_j)

if nargin == 5
	bkSize = size(IRef);
	bk_i = 1;
	bk_j = 1;
end

% get shorthands (R, t)
T = se3Exp(xi);
R = T(1:3, 1:3);
t = T(1:3,4);
RKInv = R * K^-1;

% ========= warp pixels into other image, save intermediate results ===============
% these contain the x,y image coordinates of the respective
% reference-pixel, transformed & projected into the new image.
xImg = zeros(1,numel(IRef))-10;
yImg = zeros(1,numel(IRef))-10;

% these contain the 3d position of the transformed point
xp = NaN(1,numel(IRef));
yp = NaN(1,numel(IRef));
zp = NaN(1,numel(IRef));

u = 1:size(IRef,1);
v = 1:size(IRef,2);

x = v + bkSize(2)*(bk_j-1);
y = u + bkSize(1)*(bk_i-1);

[X, Y] = meshgrid(x, y);
X = X(:).';
Y = Y(:).';

p = [X-1; Y-1; ones(size(X))] .* DRef(:).';
pTrans = RKInv * p + t;

valid_idx = find(pTrans(3,:) > 0 & transpose(DRef(:)) > 0);

% projected point (for interpolation of intensity and gradients)
pTransProj = K * pTrans;
xImg(valid_idx) = pTransProj(1,valid_idx) ./ pTransProj(3,valid_idx) - bkSize(2)*(bk_j-1);
yImg(valid_idx) = pTransProj(2,valid_idx) ./ pTransProj(3,valid_idx) - bkSize(1)*(bk_i-1);

% warped 3d point, for calculation of Jacobian.
xp(valid_idx) = pTrans(1,valid_idx);
yp(valid_idx) = pTrans(2,valid_idx);
zp(valid_idx) = pTrans(3,valid_idx);

xImg = reshape(xImg, size(IRef));
yImg = reshape(yImg, size(IRef));

xp = reshape(xp, size(IRef));
yp = reshape(yp, size(IRef));
zp = reshape(zp, size(IRef));

% interpolation
Iwarp = mirt2D_mexinterp(I, xImg+1, yImg+1);
