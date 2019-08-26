function dist = calculate_scale_error(obj, scale, P1, P1_ref)

% n = length(idx1);
% P1_ref = zeros(3, n);
% P1 = zeros(3, n);
% for i = 1:n
% 	T = obj.features{i}.transform_to_step;
% 	P1_ref(:,i) = T(1:3,:) * obj.features{idx1(i)}.point_init;
% end
% for i = 1:n
% 	P1(:,i) = obj.features{idx2(i)}.point(1:3);
% end

P1_scaled = scale * P1;
dist = ( sum( (P1_ref - P1_scaled).^2 ) ).^0.5;