function [list, nFiles] = openFiles(directory, format)
%
%
if nargin == 1
	format = {'jpg', 'jpeg', 'png', 'bmp'};
else
	format = {format};
end

if directory(end) ~= '/'
	directory = [directory '/'];
end

slist = dir(directory);
nFiles = length(slist);

removal_index = []; 
for i = 1:nFiles
	ext = extension(lower(slist(i).name));
	if slist(i).name(1) == '.' || ~(~isempty(ext) && ismember(ext, format))
		removal_index = [removal_index i];
	end
	slist(i).name = [directory slist(i).name];
end
index = 1:nFiles;
index(removal_index) = [];

slist = slist(index);
nFiles = length(slist);

list = cell(nFiles, 1);
for i = 1:nFiles
	list{i} = slist(i).name;
end