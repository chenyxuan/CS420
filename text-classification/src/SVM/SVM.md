	
	hyperplane: wx + b = 0
	we hope to maxmize r = 2 / ||w||

	max{w, b} 2 / ||w||
	s.t. yi * (w * xi + b) >= 1, i = 1, 2, ..., m.

	min{w, b} ||w||^2 / 2
	s.t. yi * (w * xi + b) >= 1, i = 1, 2, ..., m.

	L(w, b, a) = ||w||^2 / 2 + sigma {ai * (1 - yi * (w * xi + b))}

	==> w = sigma ai * yi * xi, 0 = sigma ai * yi

	max{a} sigma {ai} - 1 / 2 * sigma {sigma {ai * aj * yi * yj * xi * xj}}
	s.t. 
	sigma ai * yi =  0
	ai >= 0, i = 1, 2, ..., m

	KKT condition
	ai >= 0
	1 - yi * (w * xi + b) >= 0
	ai * (1 - yi * (w * xi + b)) = 0
	
	
	min{w, b} ||w||^2 / 2 + C * sigma ei
	s.t.
	yi * (w * xi + b) >= 1 - ei, i = 1, 2, ..., m
	ei >= 0, i = 1, 2, ..., m
	
