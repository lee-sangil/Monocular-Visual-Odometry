function h = draw_camera(h, T, rgb, axis)

persistent l w d p0 p1 p2 p3 p4 A E_add F_add L

if isempty(h)
	l=0.16; w=0.1; d=0.1;
	
	p0 = [0,0,0,1].';
	
	p1 = [w,d,l,1].';
	p2 = [-w,d,l,1].';
	p3 = [-w,-d,l,1].';
	p4 = [w,-d,l,1].';
	
	E_add = [p0,p1,p0,p2,p0,p3,p0,p4];
	F_add= [p1,p2,p3,p4,p1];
	
	L = 0.1; % coordinate axis length
	A = [0 0 0 1; L 0 0 1; 0 0 0 1; 0 L 0 1; 0 0 0 1; 0 0 L 1]';
	
end

if isempty(h)
	A_e = T*A;
	E_e = T*E_add;
	F_e = T*F_add;
	
	if axis == true
		h{1} = plot3(A_e(1,1:2),A_e(2,1:2),A_e(3,1:2),'-r','LineWidth',2); % x: red
		hold on;
		h{2} = plot3(A_e(1,3:4),A_e(2,3:4),A_e(3,3:4),'-g','LineWidth',2); % y: green
		h{3} = plot3(A_e(1,5:6),A_e(2,5:6),A_e(3,5:6),'-b','LineWidth',2); % z: blue
	end
	
	hold on;
	h{4} = plot3(E_e(1,1:2),E_e(2,1:2),E_e(3,1:2),'Color',rgb,'LineWidth',1.2);hold on
	h{5} = plot3(E_e(1,3:4),E_e(2,3:4),E_e(3,3:4),'Color',rgb,'LineWidth',1.2);
	h{6} = plot3(E_e(1,5:6),E_e(2,5:6),E_e(3,5:6),'Color',rgb,'LineWidth',1.2);
	h{7} = plot3(E_e(1,7:8),E_e(2,7:8),E_e(3,7:8),'Color',rgb,'LineWidth',1.2);
	
	h{8} = plot3(F_e(1,1:2),F_e(2,1:2),F_e(3,1:2),'Color',rgb,'LineWidth',1.5);
	h{9} = plot3(F_e(1,2:3),F_e(2,2:3),F_e(3,2:3),'Color',rgb,'LineWidth',1.5);
	h{10} = plot3(F_e(1,3:4),F_e(2,3:4),F_e(3,3:4),'Color',rgb,'LineWidth',1.5);
	h{11} = plot3(F_e(1,4:5),F_e(2,4:5),F_e(3,4:5),'Color',rgb,'LineWidth',1.5);
else
	A_e = T*A;
	E_e = T*E_add;
	F_e = T*F_add;
	
	if axis == true
		set(h{1}, 'XData', A_e(1,1:2), 'YData',A_e(2,1:2), 'ZData',A_e(3,1:2));
		set(h{2}, 'XData', A_e(1,3:4), 'YData',A_e(2,3:4), 'ZData',A_e(3,3:4));
		set(h{3}, 'XData', A_e(1,5:6), 'YData',A_e(2,5:6), 'ZData',A_e(3,5:6));
	end
	
	set(h{4}, 'XData', E_e(1,1:2), 'YData',E_e(2,1:2), 'ZData',E_e(3,1:2));
	set(h{5}, 'XData', E_e(1,3:4), 'YData',E_e(2,3:4), 'ZData',E_e(3,3:4));
	set(h{6}, 'XData', E_e(1,5:6), 'YData',E_e(2,5:6), 'ZData',E_e(3,5:6));
	set(h{7}, 'XData', E_e(1,7:8), 'YData',E_e(2,7:8), 'ZData',E_e(3,7:8));
	
	set(h{8}, 'XData', F_e(1,1:2), 'YData',F_e(2,1:2), 'ZData',F_e(3,1:2));
	set(h{9}, 'XData', F_e(1,2:3), 'YData',F_e(2,2:3), 'ZData',F_e(3,2:3));
	set(h{10}, 'XData', F_e(1,3:4), 'YData',F_e(2,3:4), 'ZData',F_e(3,3:4));
	set(h{11}, 'XData', F_e(1,4:5), 'YData',F_e(2,4:5), 'ZData',F_e(3,4:5));
end