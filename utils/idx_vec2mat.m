function varargout = idx_vec2mat(a, sz)

A(1,:) = mod(a-1, sz(1))+1;
A(2,:) = floor((a-1)/sz(1))+1;

if nargout == 1
	varargout{1} = A;
elseif nargout == 2
	varargout{1} = A(1,:);
	varargout{2} = A(2,:);
end
	