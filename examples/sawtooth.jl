m = 16
function next(x,s)
		return (2*x + s*sin(m*x)/m) %1
end
function dnext(x,s)
		return abs(2 + s*cos(m*x))
end

