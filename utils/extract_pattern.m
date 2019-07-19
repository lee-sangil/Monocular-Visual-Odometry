function des = extract_pattern(image, loc)

pattern = [  0  0;
			-1 -1;
			 1 -1;
			-1  1;
			 0  2;
			 0 -2;
			 2  0;
			-2  0;];

des = zeros(size(loc,1), 8);
for i = 1:size(loc, 1)
	des(i,:) = diag(image(loc(i,1)+pattern(:,1), loc(i,2)+pattern(:,2)));
end