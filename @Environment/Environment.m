classdef Environment < handle
	properties (Access = public)
		runMethod
	end
	properties (GetAccess = public, SetAccess = private)
		
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
				
				if ~isfield(params, 'filename')
					filename = [pkg.identifier '_' vo.identifier];
				else
					filename = params.filename;
				end
				
				params.filepath = [params.filepath char(datetime, 'yyMMdd-HHmmss/')];
				
				if ~exist(params.filepath, 'dir')
					mkdir(params.filepath);
				else
					warning('The specified folder exist. Be care for OVERWRITING.\n');
				end
				
				filename = [params.filepath filename];
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
					
					% Show image and features
					obj.plot_state(obj.plot_initialized, obj.vo, obj.pkg, obj.params);
					obj.plot_initialized = true;
					
					% Record
					if obj.params.isRecord
						for j = obj.params.figRecord
							writeVideo(obj.vidObj{j}, getframe(figure(j)));
						end
						obj.print_logs(timePassed);
					end
					
					totalTime = totalTime + timePassed;
				end
				
				disp(['total processing time: ' num2str(totalTime) ' sec, ' 'average time: ' num2str(totalTime/obj.pkg.step) ' sec']);
			end
			
		end
		
		obj = plot_state(obj, plot_initialized, vo, pkg, param)
		obj = print_logs(obj, timePassed)

	end
end