function flag = extract_features( obj )

% Update features using KLT tracker
if obj.update_features()
	
	% Delete features which is failed to track by KLT tracker
	obj.delete_dead_features();
	
	% Add features to the number of the lost features
	obj.add_features();
	
	flag = true;
	
else
	
	flag = false;

end