using CUDA
using Test
using BenchmarkTools

#next(x, s) = (2*x + s*sin(16*x)/16) % 1
#dnext(x, s) = abs(2 + s*cos(16*x))
#d2next(x, s) = -16*s*sin(16*x)
function next(x, s)
	if x < 1
		return 4*x/(1 + s + sqrt((1+s)^2 - 4*s*x))
	end
	return 4*(2 - x)/(1 + s + sqrt((1+s)^2 - 4*s*(2-x)))
end
function dnext(x, s)
	if x < 1
		pden = sqrt((1+s)^2 - 4*s*x)
		den = 1 + s + pden
		dden = 1/2/pden*(-4*s)
		term1 = 4/den 
		term2 = -4*x/den/den*dden
		return term1 + term2
	end
	pden = sqrt((1+s)^2 - 4*s*(2-x))
	den = 1 + s + pden
	dden = 1/2/pden*4*s
	term1 = -4/den 
	term2 = -4*(2-x)/den/den*dden
	return term1 + term2
end
sig_obs = 0.05
function hist_single_orbit(rho, ntime, nbins, s, a, b)
    x = rand()
	index = threadIdx().x + (blockIdx().x - 1) * blockDim().x 
    for t = 1:500
		x = next(x,s)
	end
	for t = 1:ntime
		x = next(x,s)
		bin_no = Int(cld(x, (b-a)/nbins))
		rho[(index - 1)*nbins + bin_no] += 1.0
	end    
	return nothing	
end 
function post_process_rho(rho, nbins, ntime, nsamples, nrep)
	r = zeros(nbins)
	for i = 1:nbins
		r[i] = sum(rho[i:nbins:end])/(nsamples*ntime*nrep)*nbins
	end
	return r	
end
function post_process_g(rho, nbins, ntime, nsamples, nrep)
	dlogr = zeros(nbins)
    r = post_process_rho(rho, nbins, ntime, nsamples, nrep)
	#logr = log.(r)
	logr = r
	for i = 2:nbins-1
		dlogr[i] = (logr[i+1] - logr[i-1])/2*nbins
	end
    dlogr[1] =  (logr[2] - logr[1])*nbins
    dlogr[nbins] =  (logr[nbins] - logr[nbins-1])*nbins
	return dlogr	
end

function get_emp_srb(nbins, ntime, nsamples, nrep, s, a, b)
	nrho = nsamples * nbins
	rho = CUDA.fill(0.0f0, nrho)
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)

	for n = 1:nrep
		CUDA.@sync begin
			@cuda threads=threads blocks=blocks hist_single_orbit(rho, ntime, nbins, s, a, b)
		end
	end	
	rho_final = post_process_rho(rho, nbins, ntime, nsamples, nrep)
	return rho_final
end
function bin_fo_fi(fo_sa, fo_gr, fi_gr, y, nbins, nsamples, sig, a, b)
	index = threadIdx().x + (blockIdx().x - 1)*blockDim().x

    x = fo_sa[index]

	bin_no = Int(cld(x-a, (b-a)/nbins))

	bin_no = (bin_no == 0) ? 1 : bin_no	

	ind = (index - 1)*nbins + bin_no
	fo_gr[ind] += 1
	lkhd = exp(-(y-x)^2/(2*sig*sig))
	#lkhd = abs(y-x) < sig ? 1.0 : 0.0
	fi_gr[ind] += lkhd
	return nothing
end
function post_process_fo_fi_cdfs(fo_gr, fi_gr, nbins)
	cdf = zeros(Float32, nbins)
	cdf_fo = zeros(Float32, nbins)


	for i = 1:nbins
		cdf[i] = sum(fi_gr[i:nbins:end])
		cdf_fo[i] = sum(fo_gr[i:nbins:end])
	end

	cdf = cumsum(cdf)/sum(cdf)
	cdf_fo = cumsum(cdf_fo)/sum(cdf_fo)
	return cdf, cdf_fo
end

function get_fo_fi_cdfs(fo_sa, y, nsamples, nbins, a, b)
	fo_gr = CUDA.fill(0.0f0, nbins*nsamples) 
	fi_gr = CUDA.fill(0.0f0, nbins*nsamples)
		
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)
	sig = sig_obs
	CUDA.@sync begin
			@cuda threads=threads blocks=blocks bin_fo_fi(fo_sa, fo_gr, fi_gr, y, nbins, nsamples, sig, a, b)
	end
	cdf, cdf_fo = post_process_fo_fi_cdfs(fo_gr, fi_gr, nbins)
end
function driver_cdfs(y, nsamples, nbins)
	fo_sa = CUDA.rand(nsamples)
    cdf, cdf_fo = get_fo_fi_cdfs(fo_sa, y, nsamples, nbins, a, b)
	return cdf, cdf_fo
end
function tran_lin_interp(fo_sa, fi_sa, cdf, cdf_fo, nbins, a, b)
	index = threadIdx().x + (blockIdx().x - 1)*blockDim().x
	x = fo_sa[index]
	dx = (b-a)/nbins
	i2f = Int(cld(x-a, dx))
	i2f = (i2f == 0) ? 1 : i2f
    x_mid = (i2f - 1)*dx + 0.5*dx + a
	Tx = x
	x1f, x2f = a, b
	c1f, c2f = 0.0, 1.0
	cf = 0.0
	if x > x_mid  
		x1f = x_mid
		c1f = cdf_fo[i2f]
		if !(i2f==nbins)
			x2f = x_mid + dx
			c2f = cdf_fo[i2f + 1]
		end
	else
		x2f = x_mid
		c2f = cdf_fo[i2f]
		if !(i2f==1)
			x1f = max(x_mid - dx, 0)
			c1f = (i2f > 1) ? cdf_fo[i2f-1] : 0.0
		end 
	end
	cf = c1f + (c2f - c1f)*(x - x1f)/(x2f - x1f)
	i2 = 1
	for k = 1:nbins
		if (cdf[k] >= cf)
			i2 = k
			break
		end
	end
	x2 = (i2 - 1)*dx + 0.5*dx + a
	x1 = max(x2 - dx, a)  
    c1 = (i2 > 1) ? cdf[i2 - 1] : 0.0
	c2 = cdf[i2]
	Tx = (i2 > 1) ? (cf - c1)*(x2 - x1)/(c2 - c1) + x1 : x
	fi_sa[index] = Tx
 	return nothing
end
function transport(fo_sa, y, nsamples, nbins, a, b)
	fi_sa = CUDA.fill(0.0f0, nsamples)
	cdf_temp, cdf_fo_temp = get_fo_fi_cdfs(fo_sa, y, nsamples, nbins, a, b)	
	cdf = CuArray(cdf_temp)
	cdf_fo = CuArray(cdf_fo_temp)
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)
	CUDA.@sync begin
		@cuda threads=threads blocks=blocks tran_lin_interp(fo_sa, fi_sa, cdf, cdf_fo, nbins, a, b)
	end
	return fi_sa	
end
function evolve_dynamics(samples, ntime, s)
	index = threadIdx().x + (blockIdx().x - 1)*blockDim().x
	x = samples[index]
	for i = 1:ntime
		x = next(x,s)
	end
	samples[index] = x
	return nothing
end

function get_srb_samples(nsamples, ntime, s)
	samples = CUDA.rand(Float32, nsamples) 
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)
	CUDA.@sync begin
		@cuda threads=threads blocks=blocks evolve_dynamics(samples, ntime, s)
	end
	return samples
end
function forecast(samples, nsamples, ntime, s)
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)
	CUDA.@sync begin
		@cuda threads=threads blocks=blocks evolve_dynamics(samples, ntime, s)
	end
	return nothing
end
function transport_filter(y, nsamples, ntime, ndtime, nbins, s, a, b)
	x = get_srb_samples(nsamples, 100, s)
	Sx = CUDA.fill(0.0f0, nsamples)
	for t = 1:ntime
		forecast(x, nsamples, ndtime, s)
		if t==ntime
			Sx .= x
		end
		x .= transport(x, y[t], nsamples, nbins, a, b)
	end
	return x, Sx
end
function compute_ddynamics(x, s, dx)
	index = threadIdx().x + (blockIdx().x - 1)*blockDim().x
	dx[index] = dnext(x[index], s)
	return nothing 
end
function get_dx(x, dx, nsamples, s)
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)
	CUDA.@sync begin
		@cuda threads=threads blocks=blocks compute_ddynamics(x, s, dx)
	end
	return nothing
end
function put_sample_in_bin(x, rho, dx, nbins, a)
	index = threadIdx().x + (blockIdx().x - 1)*blockDim().x
	binx = Int(cld(x[index]-a, dx))
	rho[binx + (index-1)*nbins] += 1  
	return nothing
end
function collate_rho(rho, rho_big, nsamples, nbins, dx)
	index = threadIdx().x + (blockIdx().x - 1)*blockDim().x
	for i = 1:nsamples
		rho[index] += rho_big[(i-1)*nbins + index]/nsamples/dx
	end
	return nothing
end
function bin_samples(x, rho, nsamples, nbins, a, b)
	rho_unfolded = CUDA.fill(0.0f0, nsamples*nbins)
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)
	dx = (b-a)/nbins
	CUDA.@sync begin
		@cuda threads=threads blocks=blocks put_sample_in_bin(x, rho_unfolded, dx, nbins, a)
	end
	threads = min(nbins, 1024)
	blocks = cld(nbins, threads)
	CUDA.@sync begin
		@cuda threads=threads blocks=blocks collate_rho(rho, rho_unfolded, nsamples, nbins, dx)
	end
	return nothing
end
function transport_filter_test(y, nsamples, ntime, ndtime, nbins, s, a, b)
	x = get_srb_samples(nsamples, 100, s)
	Sx = CUDA.fill(0.0f0, nsamples)
	rho_T = CUDA.fill(0.0f0, nbins)
	rho_Tm1 = CUDA.fill(0.0f0, nbins)
	for t = 1:ntime
		if t==ntime
			Sx .= x
			a1, b1 = CUDA.minimum(Sx), CUDA.maximum(Sx)
			bin_samples(Sx, rho_Tm1, nsamples, nbins, a1, b1)
		end
		forecast(x, nsamples, ndtime, s)
		a1, b1 = CUDA.minimum(x), CUDA.maximum(x)
		x .= transport(x, y[t], nsamples, nbins, a1, b1)
		println("At time ", t, " samples are in [", CUDA.minimum(x),", ", CUDA.maximum(x),"]") 
	end
	a2, b2 = CUDA.minimum(x), CUDA.maximum(x)
	bin_samples(x, rho_T, nsamples, nbins, a2, b2)
	return x, Sx, rho_Tm1, rho_T
end
function generate_obs(ntime, s)
	x = rand()
	y = zeros(ntime)
	for i = 1:ntime
		x = next(x, s)
		y[i] = x + sig_obs*randn()
	end
	return y
end
