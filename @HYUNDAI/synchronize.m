function [idx1, idx2] = synchronize(obj, t1, t2)

i = 1;
j = 1;

flag = false;
t = union(t1, t2);
id1 = ismember(t, t1);
id2 = ismember(t, t2);
idx1 = [];
idx2 = [];
for it = 1:length(t)
	
	if id1(it)
		if ~flag
			idx1 = [idx1 i];
			flag = true;
		end
		i = i + 1;
	end
	if id2(it)
		if flag
			idx2 = [idx2 j];
			flag = false;
		end
		j = j + 1;
	end
end

if flag
	idx1 = idx1(1:end-1);
end