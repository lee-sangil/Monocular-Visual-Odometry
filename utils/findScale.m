function scale = findScale(NAME, RESULT_DIR, raw_time_data, vicon, startIdx, endIdx, INIT, ENDP)

N = 10;
len = ENDP-INIT+1;
vicon_sync = zeros(7,len);

for j = 1:len
	k = round( (endIdx-startIdx)/len*(j-1)+startIdx );
	vicon_sync(:,j) = vicon(:,k); % tx ty tz qw qx qy qz
	
end

trmse = @(x,y,z)sqrt( mean( (x(1:N)-vicon_sync(1,1:N)).^2 + ...
							(y(1:N)-vicon_sync(2,1:N)).^2 + ...
							(z(1:N)-vicon_sync(3,1:N)).^2 ) );

switch (NAME)
	case 'DSO'
		FILE = load([RESULT_DIR 'dso\dso.mat']);
		
	case 'SVO'
		FILE = fopen([RESULT_DIR 'svo\10\CamTrajectory.txt']);
		C = textscan(FILE, '%f %f %f %f %f %f %f %f', 'Headerlines', 1);
		p_b = [C{:}].';
		p_b = p_b(2:end,:);
		p_b(4:7,:) = p_b([7,4,5,6],:);
		
		idx = find(C{1} == raw_time_data(INIT));
		p_b = p_b(:,idx:idx+ENDP-INIT);
		
		init_R = convert_q2r(p_b(4:7,1));
		init_t = p_b(1:3,1);
		init_tform = [init_R init_t; 0 0 0 1];
		
		for j = 1:ENDP-INIT+1
			R = convert_q2r(p_b(4:7,j));
			t = p_b(1:3,j);
			tform = [R t; 0 0 0 1];
			tform = tform / init_tform;
			
			t = tform(1:3,4);
			q = convert_r2q(tform(1:3,1:3));
			p_b(:,j) = [t;q];
		end
		
	case 'Joint-VO-SF'
		FILE = fopen([RESULT_DIR 'Joint-VO-SF\CameraTrajectory.txt']);
		C = textscan(FILE, '%f %f %f %f %f %f %f %f');
		p_b = [C{:}].';
		p_b = p_b(2:end,:);
		p_b(4:7,:) = p_b([7,4,5,6],:);
		
		idx = INIT-1;
		p_b = p_b(:,idx:idx+ENDP-INIT);
		
		init_R = convert_q2r(p_b(4:7,1));
		init_t = p_b(1:3,1);
		init_tform = [init_R init_t; 0 0 0 1];
		
		for j = 1:ENDP-INIT+1
			R = convert_q2r(p_b(4:7,j));
			t = p_b(1:3,j);
			tform = [R t; 0 0 0 1];
			tform = tform / init_tform;
			
			t = tform(1:3,4);
			q = convert_r2q(tform(1:3,1:3));
			p_b(:,j) = [t;q];
		end
end

prev_rmse = inf;
for scale = 0.5:0.01:6
	switch (NAME)
		case 'DSO'
			if isempty(FILE.abs_tform{INIT})
				init_tform = eye(4);
			else
				init_tform = FILE.abs_tform{INIT};
				init_tform(1:3,4) = scale*init_tform(1:3,4);
			end
			
			p_c = zeros(7,1);
			abs_tform = eye(4);
			prev_tform = [];
			for j = INIT:ENDP
				tform = FILE.abs_tform{j};
				if isempty(tform)
					if isempty(prev_tform)
						tform = eye(4);
					else
						tform = prev_tform;
						abs_tform = prev_tform * abs_tform;
					end
					prev_tform = [];
				else
					tform(1:3,4) = scale*tform(1:3,4);
					prev_tform = tform;
				end
				
				tform = tform*abs_tform;
				
				P = tform / init_tform;
				p_c(1:3,j-INIT+1) = P(1:3,4);
				p_c(4:7,j-INIT+1) = convert_r2q(P(1:3,1:3)); % w x y z
			end
			
		case 'SVO'
			
			p_c(1:3,:) = scale * p_b(1:3,:);
			
		case 'Joint-VO-SF'
			
			p_c(1:3,:) = scale * p_b(1:3,:);
	end
	
	rmse = trmse(p_c(1,:), p_c(2,:), p_c(3,:));
	if rmse > prev_rmse
		scale = scale-0.01;
		break;
	end
	prev_rmse = rmse;
end

fclose all;