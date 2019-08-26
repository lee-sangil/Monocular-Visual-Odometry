function ext = extension(name)
%
%
C = strsplit(name, '.');
if length(C)==1
	ext = [];
else
	ext = C{end};
end

end