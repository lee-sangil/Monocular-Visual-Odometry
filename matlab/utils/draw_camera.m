function h = draw_camera(h, T, rgb, axis, ratio)

if isempty(h)
	
	if nargin == 4
		ratio = 1;
	end
	
	l = 0.16*ratio;
	w = 0.1*ratio;
	d = 0.1*ratio;
	L = 0.1*ratio; % coordinate axis length
	
	p0 = [0,0,0,1].';
	p1 = [w,d,l,1].';
	p2 = [-w,d,l,1].';
	p3 = [-w,-d,l,1].';
	p4 = [w,-d,l,1].';
	
	E_add = [p0,p1,p0,p2,p0,p3,p0,p4];
	F_add = [p1,p2,p3,p4,p1];
	A = [0 0 0 1; L 0 0 1; 0 0 0 1; 0 L 0 1; 0 0 0 1; 0 0 L 1]';
	
	h.A = A;
	h.E_add = E_add;
	h.F_add = F_add;
	
	A_e = T*A;
	E_e = T*E_add;
	F_e = T*F_add;
	
	if axis == true
		h.axis(1) = plot3(A_e(1,1:2),A_e(2,1:2),A_e(3,1:2),'-r','LineWidth',2); % x: red
		hold on;
		h.axis(2) = plot3(A_e(1,3:4),A_e(2,3:4),A_e(3,3:4),'-g','LineWidth',2); % y: green
		h.axis(3) = plot3(A_e(1,5:6),A_e(2,5:6),A_e(3,5:6),'-b','LineWidth',2); % z: blue
		h.draw_axis = true;
	else
		h.draw_axis = false;
	end
	
	if ~isempty(rgb)
		hold on;
		h.cam(1) = plot3(E_e(1,1:2),E_e(2,1:2),E_e(3,1:2),'Color',rgb,'LineWidth',1.2);hold on
		h.cam(2) = plot3(E_e(1,3:4),E_e(2,3:4),E_e(3,3:4),'Color',rgb,'LineWidth',1.2);
		h.cam(3) = plot3(E_e(1,5:6),E_e(2,5:6),E_e(3,5:6),'Color',rgb,'LineWidth',1.2);
		h.cam(4) = plot3(E_e(1,7:8),E_e(2,7:8),E_e(3,7:8),'Color',rgb,'LineWidth',1.2);
		
		h.cam(5) = plot3(F_e(1,1:2),F_e(2,1:2),F_e(3,1:2),'Color',rgb,'LineWidth',1.5);
		h.cam(6) = plot3(F_e(1,2:3),F_e(2,2:3),F_e(3,2:3),'Color',rgb,'LineWidth',1.5);
		h.cam(7) = plot3(F_e(1,3:4),F_e(2,3:4),F_e(3,3:4),'Color',rgb,'LineWidth',1.5);
		h.cam(8) = plot3(F_e(1,4:5),F_e(2,4:5),F_e(3,4:5),'Color',rgb,'LineWidth',1.5);
		
		h.draw_skel = true;
	else
		h.draw_skel = false;
	end
else
	if nargin > 2
		warning('Too many input arguments');
	end
	
	A_e = T*h.A;
	E_e = T*h.E_add;
	F_e = T*h.F_add;
	
	if h.draw_axis
		set(h.axis(1), 'XData', A_e(1,1:2), 'YData',A_e(2,1:2), 'ZData',A_e(3,1:2));
		set(h.axis(2), 'XData', A_e(1,3:4), 'YData',A_e(2,3:4), 'ZData',A_e(3,3:4));
		set(h.axis(3), 'XData', A_e(1,5:6), 'YData',A_e(2,5:6), 'ZData',A_e(3,5:6));
	end
	
	if h.draw_skel
		set(h.cam(1), 'XData', E_e(1,1:2), 'YData',E_e(2,1:2), 'ZData',E_e(3,1:2));
		set(h.cam(2), 'XData', E_e(1,3:4), 'YData',E_e(2,3:4), 'ZData',E_e(3,3:4));
		set(h.cam(3), 'XData', E_e(1,5:6), 'YData',E_e(2,5:6), 'ZData',E_e(3,5:6));
		set(h.cam(4), 'XData', E_e(1,7:8), 'YData',E_e(2,7:8), 'ZData',E_e(3,7:8));
		
		set(h.cam(5), 'XData', F_e(1,1:2), 'YData',F_e(2,1:2), 'ZData',F_e(3,1:2));
		set(h.cam(6), 'XData', F_e(1,2:3), 'YData',F_e(2,2:3), 'ZData',F_e(3,2:3));
		set(h.cam(7), 'XData', F_e(1,3:4), 'YData',F_e(2,3:4), 'ZData',F_e(3,3:4));
		set(h.cam(8), 'XData', F_e(1,4:5), 'YData',F_e(2,4:5), 'ZData',F_e(3,4:5));
	end
end