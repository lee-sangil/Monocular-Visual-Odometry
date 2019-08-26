function pt = randidx(pts, mask, weight)
% idx = randidx(pts, mask, weight)
% A point will be chosen with high probability, if its weight is high

if size(mask,2) == 1
	mask = transpose(mask);
end

if size(weight,2) == 1
	weight = transpose(weight);
end

if size(pts,2) ~= size(mask,2) || size(pts,2) ~= size(weight,2)
	error('the size of vector is unmatched.');
end

weight(isnan(weight)) = 0;
if sum(weight) == 0
	weight = ones(size(weight));
end
	
I = find(mask);
weight = weight(I);

select = find( rand(1) <= cumsum(weight./sum(weight)), 1, 'first');
pt = pts(:,I(select));