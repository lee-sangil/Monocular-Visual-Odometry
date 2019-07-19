function [idx, length] = seek_index( obj, max_length, condition )

idx_t = zeros(1, max_length);
j = 1;
for i = 1:obj.nFeature
	if condition(i)
		idx_t(j) = i;
		j = j + 1;
	end
end
length = j-1;
idx = idx_t(1:length);