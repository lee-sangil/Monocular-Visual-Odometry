function flag = extract_features( obj )

%
if obj.update_features()
	%
	obj.delete_dead_features();
	
	%
	obj.add_features();
	
	flag = true;
else
	
	flag = false;

end