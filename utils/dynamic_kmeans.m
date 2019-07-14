function [idx, C, bin] = dynamic_kmeans(X, varargin)
%dynamic_kmeans Advanced K-means clustering
%   [idx, C, bin] = dynamic_kmeans(X, varargin)

param = parse_varargin(varargin);

% Constant
n = size(X, 2);
idx = nan(1, n);

% Iterative parameter
remain_idx = 1:n;
k = 0;

% Do-iteration
while ~isempty(remain_idx)
	k = k + 1;
	
	i_sel = remain_idx(randi(length(remain_idx), 1));
	C(:,k) = X(:, i_sel);
	
	while true
		
		idx_temp = [];
		for i = remain_idx
			dist = param.method(X(:,i) - C(:,k));
			if dist < param.maxDist
				idx_temp(i) = k;
			end
		end
		
		C_prev = C(:,k);
		C(:,k) = mean(X(:,idx_temp==k),2);
		
		if C(:,k) == C_prev
			idx(idx_temp==k) = k;
			break;
		end
		
	end
	
	remain_idx = find(isnan(idx) == 1);
end

bin = histcounts(idx, (0:k)+0.5);

% Sort by the number of group
if param.isSort
	[~, sIdx] = sort(bin, param.sortDir);
	
	% Re-arrange
	idx_sort = zeros(size(idx));
	for j = 1:k
		idx_sort(idx==sIdx(j)) = j;
	end
	
	idx = idx_sort;
	bin = histcounts(idx, (0:k)+0.5);
end

end

function param = parse_varargin(input)

% Default
param.isSort = false;
param.sortDir = 'descend';
param.maxDist = 1;
param.method = @norm;

% Parse
for i = 1:2:length(input)
	switch lower(input{i})
		case 'maxdist'
			param.maxDist = input{i+1};
		case 'method'
			param.method = input{i+1};
		case 'sortdir'
			if strcmpi(input{i+1}, 'ascend') || strcmpi(input{i+1}, 'descend')
				param.sortDir = input{i+1};
				param.isSort = true;
			else
				warning('SortDir should be ''ascend'' or ''descend''. A default option is not to sort.');
			end
	end
end

end