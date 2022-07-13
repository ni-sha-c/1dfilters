function resample(x,w)
    cdfw = cumsum(w)
    new_pts = similar(x)
	N_p = size(x)[1]
    for j = 1:N_p
        r = rand()
        for i = 1:N_p
		    if cdfw[i] >= r
                new_pts[j] = x[i]
			    break
			end
		end
    end
	return new_pts
end
function analysis(w, y, x)
    N_p = size(x)[1]
	lw = log.(w)
	for k = 1:N_p
		lw[k] = lw[k] + log_likelihood(y - x[k]) 
	end
	w1 = exp.(lw)
	w1 ./= sum(w1)
	return w1
end

function sir(x,y,w,τ,T,N,N_thr,s)
	x_pr = zeros(N,T)
	w_pr = zeros(N,T)
	for t = 1:T
		x .= forecast(x, s, τ)
		w .= analysis(w, y[t], x)
		N_eff = 1.0/sum(w.*w)
        if (N_eff < N_thr)
			x .= resample(x,w)
	    end
		x_pr[:,t] .= x
		w_pr[:,t] .= w
    end
	return x_pr, w_pr
end
function assimilate(N, Nthr, τ, T, s)
	x_t, y = generate_data(T, s)
	x = rand(N)
	x .= forecast(x, s, 1000)
	w = ones(N)/N
	x_pr, w_pr = sir(x,y,w,τ,T,N,Nthr,s)
	return x_pr, w_pr, x_t, y 
end



