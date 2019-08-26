function S = yamlread(filename)

fid = fopen(filename, 'r');
if fid == -1
	error('cannot open the file');
end

i = 0;

tline = fgetl(fid);
while ischar(tline)
	
    comment = find(tline == '#', 1);
	version = find(tline == '%', 1);
	name_back = find(tline == ':');
	name_front = 1;
	value_front = 1;
	
	if ~isempty(comment) || ~isempty(version)
		flag = false;
		
	else
		if ~isempty(name_back)
			i = i + 1;
			
			value_front = name_back+1;
			
			name{i} = tline(name_front:name_back-1);
			value{i} = [];
			
			flag = true;
			
		end
		
		if flag
			
			rows = cell2mat(cellfun(@str2num, strsplit(tline(value_front:end), {',', ' '}), 'un', 0));
			value{i} = [value{i}; rows];
			
		end
		
	end
	
    tline = fgetl(fid);
end

for j = 1:i
	eval(sprintf('S.%s=value{%d};', name{j}, j));
end

% command = 'S = struct(';
% for j = 1:i
% 	if j > 1
% 		command = [command ', '];
% 	end
% 	command = [command sprintf('''%s'', value{%d}', name{j}, j)];
% end
% command = [command, ');'];
% 
% eval(command);

fclose(fid);