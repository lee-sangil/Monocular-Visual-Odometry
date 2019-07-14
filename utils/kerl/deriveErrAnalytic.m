function [ Jac, residual ] = deriveErrAnalytic( IRef, DRef, I, xi, K )

[Iwarp, xImg, yImg, xp, yp, zp] = transform_kerl(IRef, DRef, I, xi, K);

% ========= calculate actual derivative. ===============
% 1.: calculate image derivatives, and interpolate at warped positions.
dxI = NaN(size(I));
dyI = NaN(size(I));
dyI(2:(end-1),:) = 0.5*(I(3:(end),:) - I(1:(end-2),:));
dxI(:,2:(end-1)) = 0.5*(I(:,3:(end)) - I(:,1:(end-2)));
dxInterp = K(1,1) * reshape(mirt2D_mexinterp(dxI, xImg+1, yImg+1),size(I,1) * size(I,2),1);
dyInterp = K(2,2) * reshape(mirt2D_mexinterp(dyI, xImg+1, yImg+1),size(I,1) * size(I,2),1);

% 2.: get warped 3d points (x', y', z').
xp = reshape(xp,size(I,1) * size(I,2),1);
yp = reshape(yp,size(I,1) * size(I,2),1);
zp = reshape(zp,size(I,1) * size(I,2),1);

% 3. direct implementation of kerl2012msc.pdf Eq. (4.14):
Jac = zeros(size(I,1) * size(I,2),6);
Jac(:,1) = dxInterp ./ zp;
Jac(:,2) = dyInterp ./ zp;
Jac(:,3) = - (dxInterp .* xp + dyInterp .* yp) ./ (zp .* zp);
Jac(:,4) = - (dxInterp .* xp .* yp) ./ (zp .* zp) - dyInterp .* (1 + (yp ./ zp).^2);
Jac(:,5) = + dxInterp .* (1 + (xp ./ zp).^2) + (dyInterp .* xp .* yp) ./ (zp .* zp);
Jac(:,6) = (- dxInterp .* yp + dyInterp .* xp) ./ zp;
% invert jacobian: in kerl2012msc.pdf, the difference is defined the other
% way round, see (4.6).
Jac = -Jac;
residual = reshape(IRef-Iwarp,size(I,1) * size(I,2),1);

% ========= plot residual image =========
figure(6)
imagesc(reshape(residual,size(I)));
colormap gray;
drawnow;
set(gca, 'CLim', [-1,1]);
end

