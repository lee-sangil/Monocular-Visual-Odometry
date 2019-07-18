function dist = essential_model_error(obj, E, x1, x2, weight)

% sampson error
Ex1 = E*x1;
Etx2 = E.'*x2;

% for i = 1:n
% 	d(i) = ( uv2(:,i).'*F*uv1(:,i) ).^2 ./ ( Fu1(i)^2 + Fv1(i)^2 + Ftu2(i)^2 + Ftv2(i)^2 );
% end

% d = diag(uv2.'*F*uv1).^2 ./ sum( (F(1:2,:)*uv1).^2 + (F(:,1:2).'*uv2).^2 ).';

dist = weight .* diag(x2.'*E*x1).^2 ./ ( Ex1(1,:).^2 + Ex1(2,:).^2 + Etx2(1,:).^2 + Etx2(2,:).^2 + eps ).';