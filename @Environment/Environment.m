classdef Environment < handle
	properties (Access = public)
		runMethod
	end
	properties (Access = private)
		
		isInitialized = true
		plot_initialized = false
		vo
		pkg
		vidObj
		params
		kb
		
	end
	methods (Access = public)
		function obj = Environment(vo, pkg, params)
			
			obj.vo = vo;
			obj.pkg = pkg;
			obj.params = params;
			
			if obj.params.isRecord
				% Designate filename
				if ~isfield(params, 'filepath')
					params.filepath = 'logs/';
				end
				
				if ~exist(params.filepath, 'dir')
					mkdir(params.filepath);
				else
					warning('The specified folder exist. Be care for OVERWRITING.\n');
				end
				
				if ~isfield(params, 'filename')
					filename = [params.filepath pkg.identifier '_' vo.identifier];
				else
					filename = [params.filepath params.filename];
				end
				
				filename = [filename char(datetime, '-yyMMdd-hhmmss')];
				filename_ = cell(length(obj.params.figRecord));
				for j = obj.params.figRecord
					filename_{j} = [filename '-' num2str(j)];
				end
				
				if ~isempty(obj.params.figRecord)
					temp = char('0'+obj.params.figRecord(1));
					for j = obj.params.figRecord(2:end)
						temp = [temp ',' num2str(j)];
					end
					disp(['RECORD: ' filename '-{' temp '}.avi']);
				end
				
				% Ask the user for confirmation
				c = input('press ENTER to continue or Q to quit.', 's');
				if ~isempty(c) && (c(end) == 'q' || 'Q')
					obj.isInitialized = false;
					return;
				end
				
				% Construct video object
				for j = obj.params.figRecord
					obj.vidObj{j} = VideoWriter(filename_{j}, 'Motion JPEG AVI');
					set(obj.vidObj{j}, 'FrameRate', 20);
					open(obj.vidObj{j});
				end
				
				obj.params.filepath = params.filepath;
			end
			
			% Construct keyboard object
			HebiKeyboard.loadLibs();
			obj.kb = HebiKeyboard();
			
		end
		
		function obj = delete( obj )
			
			% Delete video and keyboard object
			if obj.isInitialized
				if obj.params.isRecord
					for j = obj.params.figRecord
						close(obj.vidObj{j});
					end
				end
				
				close(obj.kb);
			end
		end
		
		function obj = run( obj )
			
			totalTime = 0;
			if obj.isInitialized
				while ~end_of_file(obj.pkg)
					
					% Read keys
					pause(0.01);
					state = read(obj.kb);
					
					if any(state.keys(['q', 'Q']))
						break;
					end
					
					% VO run
					tic;
					obj.runMethod(obj.pkg);
					timePassed = toc;
					
					if obj.vo.processed
						% Show image and features
						obj.plot_state(obj.plot_initialized, obj.vo, obj.pkg, obj.params);
						obj.plot_initialized = true;
						
						% Record plot
						if obj.params.isRecord
							for j = obj.params.figRecord
								writeVideo(obj.vidObj{j}, getframe(figure(j)));
							end
						end
						
					end
				end
				
				switch obj.vo.identifier
					case 'sparseflow'
						figure();
						obj.vo.plot_spacetime();
						
						% SAVE
						if obj.params.isRecord
							save([obj.params.filepath obj.pkg.get_identifier '.mat']);
							odometry = obj.vo.get_quaternion;
							odometry = odometry(:,:,1);
							save([obj.params.filepath 'QocRec.mat'], 'odometry');
						end
						
					case 'rgbd'
						if obj.params.isRecord
							save([obj.params.filepath obj.pkg.get_identifier '.mat']);
							TocRec = obj.vo.get_odometry;
							for i = 1:length(TocRec)
								mat = inv(TocRec{i});
								q = convert_r2q(mat(1:3,1:3));
								t = mat(1:3,4);
								odometry(:,i) = [t;q];
							end
							save([obj.params.filepath 'QocRec.mat'], 'odometry');
						end
						
					case 'stream'
						figure();
						obj.vo.print_depthHist();
				end
				
				disp(['total processing time: ' num2str(totalTime) ' sec, ' 'average time: ' num2str(totalTime/obj.pkg.step) ' sec']);
				
				if obj.params.isRecord
					fid = fopen([obj.params.filepath 'log.txt'], 'w');
					fprintf(fid, 'total processing time: %f sec, average time: %f sec', totalTime, totalTime/obj.pkg.step);
					fclose(fid);
				end
			end
			
		end
		
		obj = plotOverview( obj, plot_initialized, vo, pkg, params )
		
	end
end