using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, GraphNeuralNetworks, Statistics, LinearAlgebra, Graphs, SparseArrays, Juno, ForwardDiff, Printf, ProgressBars, Distributions, FFTW, ReverseDiff, FileIO, JLD2, Optimisers

include("MyDeepONet.jl")
using .MyDeepONet


# Data setup
xi = -0.5
xf = 0.5
ti = -0.5
tf = 0.5
yspan = [xi xf;ti tf]
D=0.05#10000  # Diffusivity
vel=1  # Velocity
n_u_trajectories = 1000
n_u_trajectories_test = 1000
n_u_trajectories_validation = 1000
n_y_eval = 200
batch_size = 50
n_epochs = 200
n_freq_fft = 200
n_freq_gen = 100
n_spatial_finite_diff = 200
frequency_decay = 0.25  # Larger number means faster decay, meaning fewer high frequency components
Random.seed!(0)
flux_ini = Flux.glorot_uniform(MersenneTwister(rand(Int64)))
recompute_data = false
save_on_recompute = true
training_var_time = true
const_bias_trainable = false
trunk_var_bias = true
equidistant_y = false

# Model setup
if length(ARGS) == 0
    n_sensors = 100
    branch_width = 75
    trunk_width = 75
    latent_size = 100
    activation_function = softplus
    branch_depth = 4
    trunk_depth = 4
    physics_weight_initial = 0.0
    physics_weight_boundary = 0.0
    physics_weight_interior = 1.0
    data_weight = 0.0
    regularisation_weight = 0.0
    PI_use_AD = false  # AD not currently working
    do_plots = true
else
    jobindex = parse(Int64, ARGS[1])

    # (n_sensors,branch_width,trunk_width,latent_size,activation_function,branch_depth,trunk_depth,physics_weight) = [
    # (n_sensors,width,width,latent_width,activation_function,depth,depth,physics_weight)
    # for n_sensors in [50,100,150]
    # for width in [50,75,100]
    # for latent_width in [50,75,100]
    # for activation_function in [softplus,tanh,sigmoid]
    # for depth in [3,4,5]
    # for physics_weight in [0.1,0.5,1.0]
    # ][jobindex]
    #
    # physics_weight_initial = physics_weight
    # physics_weight_boundary = physics_weight
    # physics_weight_interior = physics_weight
    # data_weight = 1.0
    # regularisation_weight = 0.0


    # (physics_weight_initial,physics_weight_boundary,physics_weight_interior) = [
    # (initial, internal, boundary)
    # for initial in [0.0, 0.1, 0.2, 0.3]
    # for internal in [0.0, 0.1, 0.2, 0.3]
    # for boundary in [0.0, 0.1, 0.2, 0.3]
    # ][jobindex]
    # branch_width = 50
    # trunk_width = 50
    # branch_depth = 4
    # trunk_depth = 4
    # n_sensors = 50
    # latent_size = 75
    # activation_function = softplus
    # data_weight = 1.0
    # regularisation_weight = 0.0


    (branch_width,trunk_width,branch_depth,trunk_depth) = [
    (initial, internal, boundary)
    for bw in []
    for tw in []
    for bd in []
    for td in []
    ][jobindex]
    physics_weight_initial =
    physics_weight_boundary =
    physics_weight_interior =
    n_sensors = 50
    latent_size = 75
    activation_function = softplus
    data_weight = 1.0
    regularisation_weight = 0.0


    PI_use_AD = false  # AD not currently working
    do_plots = false
end



## For solving the pde as coupled odes in frequency domain


fft_diff_locs = range(start=xi, stop=xf, length=n_freq_fft+1)[begin:end-1]
x_locs = range(start=xi, stop=xf, length=n_sensors+1)[begin:end-1]
x_locs_full = cat(x_locs,xf,dims=1)


omega = fftfreq(n_freq_fft,1/(fft_diff_locs[2]-fft_diff_locs[1])) * 2*π
coef_vec = -D*omega.^2 .- vel*omega*im
function f_fft(fft_c, p, t)
    return coef_vec .* fft_c
end


## Define functions

base_wave_number = 2*π/(xf-xi)
wave_numbers = (0:n_freq_gen).*base_wave_number
function get_u(seed)
    # Define input function
    rng = MersenneTwister(seed)
    # A = rand(rng, Uniform(0, 10), n_freq_gen) .* sqrt.(exp.(-wave_numbers.^2 ./ (2*base_wave_number^2)))
    # A = rand(rng, Uniform(0, 10), n_freq_gen) .* exp.(-wave_numbers.^2 .* (frequency_decay / base_wave_number^2))
    A = rand(rng, Uniform(0, 10), n_freq_gen+1) .* exp.(frequency_decay * (- (0:n_freq_gen).^2))
    ϕ = rand(rng, Uniform(0, 2*π), n_freq_gen+1)

    return x->sum([A[i]*cos.(wave_numbers[i]*x.+ϕ[i]) for i in 1:n_freq_gen])
end



function v_func(yt, seed; manual_u = nothing)
    # Solve problem in frequency domain with numerical ode solver (4th order Runge-Kutta) and
    # evaluate solution at points y
    if manual_u == nothing
        u = get_u(seed)
        u_fft = fft(u(fft_diff_locs))
    else
        u_fft = fft(manual_u)
    end
    times = sort(unique(yt[2,:]))

    tspan = [ti,times[end]]
    prob = ODEProblem(f_fft, u_fft, tspan, saveat=times, tstops=times)

    v_values = real(ifft.(solve(prob, Tsit5()).u))

    cat_fft_locs = (cat(fft_diff_locs,xf, dims=1), )
    inter = [interpolate(
        cat_fft_locs,
        cat(v,v[begin],dims=1),
        Gridded(Interpolations.Linear())) for v in v_values]


    unsorted_idx = [findall(yt[2,i].==times)[] for i in 1:size(yt,2)]

    return [inter[unsorted_idx[i]](yt[1,i]) for i in 1:size(yt,2)]
end


u_func(x_locs, seed) = get_u(seed)(x_locs)



## For the finite difference, method of lines

spatial_finite_diff_locs = range(start=xi, stop=xf, length=n_spatial_finite_diff+1)[begin:end-1]
spatial_diff = spatial_finite_diff_locs[2]-spatial_finite_diff_locs[1]
# If we did not use periodic boundary, then I would need x_locs to start and end on the boundaries.

# @assert (n_spatial_finite_diff/n_sensors) % 1.0 <= 1e-8 # Check that it is close to integer, so sensor points and FD points line up
# finite_diff_to_x_locs_idx = (0:n_sensors-1)*round(Int,n_spatial_finite_diff/n_sensors) .+ 1
# x_locs_FD = spatial_finite_diff_locs[finite_diff_to_x_locs_idx] # Sensor locations (input function evaluation points)



# using LinearAlgebra, SparseArrays
# A = sparse(diagm(0 => -2*D*ones(n_spatial_finite_diff), 1 => (D-vel/2)*ones(n_spatial_finite_diff-1), -1 => (D+vel/2)*ones(n_spatial_finite_diff-1)))
# A[1,end] = D+vel/2
# A[end,1] = D-vel/2
function f_FD(c, p, t)
    n = length(c)
    dc = [
    begin
        ip1 = i==n_spatial_finite_diff ? 1 : i+1
        im1 = i==1 ? n_spatial_finite_diff : i-1
        D/spatial_diff^2 * (c[im1] -2*c[i] + c[ip1]) - vel/(2*spatial_diff) * (c[ip1]-c[im1])#-2*D*c[i] + (D-vel/2)*c[ip1] + (D+vel/2)*c[im1]
    end
    for i in 1:n]

    # above is roughly twice as fast as below with A defined as in comment above
    #dc = A * c

    return dc
end

function v_func_FD(yt, seed; manual_u = nothing)
    # Solve problem with numerical ode solver (4th order Runge-Kutta) and
    # evaluate solution at points y

    if manual_u == nothing
        u = get_u(seed)
        u_vals = u(spatial_finite_diff_locs)
    else
        u_vals = manual_u
    end
    times = sort(unique(yt[2,:]))

    tspan = [ti,times[end]]
    prob = ODEProblem(f_FD, u_vals, tspan, saveat=times, tstops=times)

    v_values = solve(prob, Tsit5()).u

    cat_FD_locs = (cat(spatial_finite_diff_locs,xf, dims=1), )
    inter = [interpolate(
        cat_FD_locs,
        cat(v,v[begin],dims=1),
        Gridded(Interpolations.Linear())) for v in v_values]


    unsorted_idx = [findall(yt[2,i].==times)[] for i in 1:size(yt,2)]

    return [inter[unsorted_idx[i]](yt[1,i]) for i in 1:size(yt,2)]
end

## Generate data
setup_hash = hash((n_sensors,yspan,D,vel,n_u_trajectories,n_u_trajectories_test,n_u_trajectories_validation,n_y_eval,batch_size,n_freq_fft,frequency_decay,training_var_time,wave_numbers))
data_filename = "convection_diffusion_var_time_data_hash_$setup_hash.jld2"

if isfile(data_filename) && !recompute_data
    loaders = FileIO.load(data_filename,"loaders")
    println("Loaded data from disk")
    flush(stdout)
else
    y_locs = generate_y_locs(yspan, n_y_eval, n_u_trajectories+n_u_trajectories_test+n_u_trajectories_validation, equidistant_y)
    if !training_var_time
        y_locs[2,:,:] .= tf
    end
    loaders = generate_data(x_locs, yspan, u_func, v_func, n_sensors, n_u_trajectories, n_u_trajectories_validation, n_u_trajectories_test, n_y_eval, batch_size; equidistant_y=equidistant_y, y_locs=y_locs)
    if save_on_recompute
        FileIO.save(data_filename,"loaders",loaders)
    end
end



## Define layers

@assert branch_depth >= 3
branch = Chain(
    Dense(n_sensors, branch_width, activation_function, init=flux_ini),
    [Dense(branch_width, branch_width, activation_function, init=flux_ini) for _ in 1:branch_depth-3]...,
    Dense(branch_width, latent_size, init=flux_ini)
)

@assert trunk_depth >= 3
if training_var_time
    trunk = Chain(
        Dense(2, trunk_width, activation_function, init=flux_ini),
        [Dense(trunk_width, trunk_width, activation_function, init=flux_ini) for _ in 1:trunk_depth-3]...,
        Dense(trunk_width, latent_size+trunk_var_bias, activation_function, init=flux_ini)
    )
else
    trunk = Chain(
        x->reshape(x[1,:],1,:),
        Dense(1, trunk_width, activation_function, init=flux_ini),
        [Dense(trunk_width, trunk_width, activation_function, init=flux_ini) for _ in 1:trunk_depth-3]...,
        Dense(trunk_width, latent_size+trunk_var_bias, activation_function, init=flux_ini)
    )
end

##
# Define model
float_type_func = f32
model = DeepONet(trunk=float_type_func(trunk), branch=float_type_func(branch), const_bias_trainable=const_bias_trainable, trunk_var_bias=trunk_var_bias, bias=float_type_func([0.0]))

##
# yt1 = 0
# u_vals1 = 0
# v_u_true1 = 0

function eval_trunk_and_combine(yy,bb,p)
    # if p==(nothing,)
    #     p=nothing
    # end
    return combine_latent(model,evaluate_trunk(model,yy,p),bb,p)
end

if PI_use_AD
    params = get_params(model)
else
    params = Flux.params(model)
end

if training_var_time
    ϵ = Float64(eps(float_type_func==f32 ? Float32 : Float64)^(1/3))
    first_deriv_compiled_tape = nothing
    second_deriv_compiled_tape = nothing
    function loss((yt, u_vals), v_y_true, p=nothing)
        if p!=nothing && length(p)==2
            p = [p;[0.0]]
        else
            p = p
        end
        global first_deriv_compiled_tape, second_deriv_compiled_tape
        # yt = cat(yt[1,:],ones(size(yt,2)),dims=2)'
        # global yt1, u_vals1, v_u_true1
        # yt1, u_vals1, v_u_true1 = yt, u_vals, v_y_true
        # yt, u_vals, v_y_true = yt1, u_vals1, v_u_true1

        # for i=1:size(yt,2)
            # # Spatial
            # d_y1_val = ForwardDiff.derivative(y1->model([y1; yt[2,i]], u_vals[:,i])[], yt[1,i])
            #
            # # Second spatial
            # d2_y1_val = ForwardDiff.derivative(d_y1 -> ForwardDiff.derivative(y1->model([y1; yt[2,i]], u_vals[:,i])[], d_y1), yt[1,i])
            #
            # # Temporal
            # d_y2_val = ForwardDiff.derivative(y2->model([yt[1,i], y2], u_vals[:,i])[], yt[2,i])
            #
            # # Interior
            # physics_loss_squared += (D * d2_y1_val - vel * d_y1_val - d_y2_val)^2
            #
            # # Initial
            # inter = interpolate(
            #     (x_locs_full,),
            #     cat(u_vals[:,i],u_vals[1,i],dims=1),
            #     Gridded(Interpolations.Linear())
            # )
            # physics_loss_squared += (model([yt[1,i] ; yspan[2,1]], u_vals[:,i])[] - inter(yt[1,i]))^2
            #
            # # Boundary
            # physics_loss_squared += (model([yspan[1,1] ; yt[2,i]], u_vals[:,i])[] - model([yspan[1,2] ; yt[2,i]], u_vals[:,i])[])^2
            #
            # # Data
            # data_loss_squared += (model(yt[:,i],u_vals[:,i])[] - v_y_true[i])^2
        # end

        # MÅSKE SKAL MODEL IKKE VÆRE GLOBAL?

        sensor_idx = rand(MersenneTwister(0),1:n_sensors,batch_size)  # Randomly select which sensors are used for initial value loss
        random_sensors = [u_vals[sensor_idx[i],i] for i in 1:batch_size]'

        b = evaluate_branch(model,u_vals,p)

        similar_ones = ones(eltype(yt),1,size(yt,2))

        t = evaluate_trunk(model,yt,p)
        t_sensors = evaluate_trunk(model,[x_locs[sensor_idx]' ; ti * similar_ones],p)
        t_left = evaluate_trunk(model,[xi * similar_ones ; yt[2,:]'],p)
        t_right = evaluate_trunk(model,[xf * similar_ones ; yt[2,:]'],p)

        preds = combine_latent(model,t,b,p)

        # cat_trunk_inputs = hcat(
        #     yt,  # regular input
        #     [random_sensors ; yspan[2,1] * ones(1,batch_size)],  # input at initial condition
        #     [yspan[1,1] * ones(1,batch_size) ; yt[2,:]'],  # input at left boundary
        #     [yspan[1,2] * ones(1,batch_size) ; yt[2,:]'],  # input at right boundary
        # )
        # cat_t = evaluate_trunk(model,cat_trunk_inputs)
        # t = cat_t[:,1:batch_size]
        # t_sensors = cat_t[:,batch_size+1:2*batch_size]
        # t_left = cat_t[:,2*batch_size+1:3*batch_size]
        # t_right = cat_t[:,3*batch_size+1:4*batch_size]
        # cat_t[:,2*batch_size+1:3*batch_size] .= cat_t[:,2*batch_size+1:3*batch_size] - cat_t[:,3*batch_size+1:4*batch_size] # left-right
        # cat_preds = combine_latent(model, cat_t[:,1:3*batch_size], hcat(b,b,b))








        # t = evaluate_trunk(model,yt)
        # t_sensors = evaluate_trunk(model,[random_sensors ; yspan[2,1] * ones(1,batch_size)])
        # t_left = evaluate_trunk(model,[yspan[1,1] * ones(1,batch_size) ; yt[2,:]'])
        # t_right = evaluate_trunk(model,[yspan[1,2] * ones(1,batch_size) ; yt[2,:]'])



        # # Spatial
        # d_y1_val = ForwardDiff.derivative(ytt->combine_latent(model,evaluate_trunk(model,ytt),b), yt)
        #
        # # Second spatial
        # d2_y1_val = 0 #ForwardDiff.derivative(d_y1 -> ForwardDiff.derivative(y1->model([y1; yt[2,i]], u_vals[:,i])[], d_y1), yt[1,i])
        #
        # # Temporal
        # d_y2_val = 0 #ForwardDiff.derivative(y2->model([yt[1,i], y2], u_vals[:,i])[], yt[2,i])




        # gradients = [Flux.gradient(ytt->combine_latent(model,evaluate_trunk(model,ytt),b[:,i])[], yt[:,i]) for i in 1:size(yt,2)]





        # second_spatial = [Flux.gradient(y->ForwardDiff.derivative(dy->combine_latent(model,evaluate_trunk(model,[dy,yt[2,i]]),b[i])[],y), yt[1,i]) for i in 1:size(yt,2)]



        #
        # # Interior
        # J=Flux.jacobian(y->combine_latent(model,evaluate_trunk(model,y),b), yt)[1]
        # Hd=[Flux.diaghessian(y->combine_latent(model,evaluate_trunk(model,[y; yt[2,i]]),b[:,i])[], yt[1,i])[1] for i=1:size(yt,2)]
        # physics_loss_squared += sum([(D * Hd[i] - vel * J[i,2*(i-1)+1] - J[i,2*i])^2 for i=1:size(yt,2)])

        # y1_deriv = [ReverseDiff.gradient(y1->combine_latent(model,evaluate_trunk(model,[y1' ; yt[2,i]]),b[:,i])[], yt[1,i]) for i=1:size(yt,2)]
        # y2_deriv = [ReverseDiff.gradient(y2->combine_latent(model,evaluate_trunk(model,[yt[1,i]; y2']),b[:,i])[], yt[2,i]) for i=1:size(yt,2)]
        # y1_2_deriv = [ReverseDiff.gradient(dy1->ReverseDiff.gradient(y1 -> combine_latent(model,evaluate_trunk(model,[y1' ; yt[2,i]]),b[:,i])[], dy1), yt[1,i]) for i=1:size(yt,2)]

        # for i in 1:size(yt,2)
        #     y1_deriv = ReverseDiff.gradient(y1->combine_latent(model,evaluate_trunk(model,[y1' ; yt[2,i]]),b[:,i])[], [yt[1,i]], cfg)[]
        #     y2_deriv = ReverseDiff.gradient(y2->combine_latent(model,evaluate_trunk(model,[yt[1,i]; y2']),b[:,i])[], [yt[2,i]], cfg)[]
        #     y1_2_deriv = ReverseDiff.gradient(dy1->ReverseDiff.gradient(y1 -> combine_latent(model,evaluate_trunk(model,[y1' ; yt[2,i]]),b[:,i])[], dy1), [yt[1,i]], cfg)[]
        #     physics_loss_squared += (D * y1_2_deriv - vel * y1_deriv - y2_deriv)^2
        # end


        # output = zeros(Float64,3,batch_size)
        # function eval_trunk_and_combine(value_output,yy,bb)
        #     value_output .= combine_latent(model,evaluate_trunk(model,yy),bb)'
        # end
        # function val_and_derivative(value_diff_output,yy,bb)
        #     (y_deriv,b_deriv) = ReverseDiff.jacobian(eval_trunk_and_combine, value_diff_output[1,:], (yy, bb))
        #     value_diff_output[2,:] .= [y_deriv[i,2*(i-1)+1] for i=1:batch_size] #y1 derivative, spatial
        #     value_diff_output[3,:] .= [y_deriv[i,2*i] for i=1:batch_size] #y2 derivative, temporal
        # end
        # y_2_deriv,_ = ReverseDiff.jacobian(val_and_derivative, output, (yt, b))



        # function eval_trunk_and_combine(yy,bb)
        #     return combine_latent(model,evaluate_trunk(model,yy),bb)
        # end
        # y_deriv = ReverseDiff.jacobian(eval_trunk_and_combine, (yt, b))[1]
        # y1_deriv = [y_deriv[i,2*(i-1)+1] for i=1:batch_size] #y1 derivative, spatial
        # y2_deriv = [y_deriv[i,2*i] for i=1:batch_size] #y2 derivative, temporal

        if PI_use_AD# && false
            J=Flux.jacobian(y->eval_trunk_and_combine(y,b,p), yt)[1]
            y1_1_deriv = [J[i,2*(i-1)+1] for i=1:size(yt,2)]
            y2_1_deriv = [J[i,2*i] for i=1:size(yt,2)]
            y1_2_deriv=[Flux.diaghessian(y->eval_trunk_and_combine([y; yt[2,i]],b[:,i],p)[], yt[1,i])[1] for i=1:size(yt,2)]

        elseif PI_use_AD && false
            if first_deriv_compiled_tape == nothing || second_deriv_compiled_tape == nothing
                input_example = (yt[:,1], b[:,1], p...)
                cfg = ReverseDiff.GradientConfig(input_example)
                first_deriv_compiled_tape = ReverseDiff.compile(ReverseDiff.GradientTape((yy,bb,pp...)->eval_trunk_and_combine(yy,bb,[pp...]), input_example, cfg))
                second_deriv_compiled_tape = ReverseDiff.compile(ReverseDiff.GradientTape((yy,bb,pp...)->ReverseDiff.gradient((yyy,bbb,ppp...)->eval_trunk_and_combine(yyy,bbb,[ppp...]),(yy,bb,pp...))[1][1], input_example, cfg))
                # second_deriv_compiled_tape = ReverseDiff.compile(ReverseDiff.GradientTape((yy,bb,p...)->ReverseDiff.gradient!(first_deriv_compiled_tape,(yy,bb,p...))[1][1], input_example, cfg))
            end


            y1_derivatives = map(1:batch_size) do i
                # deriv = ReverseDiff.gradient(eval_trunk_and_combine, (yt[:,i], b[:,i]))[1]
                return ReverseDiff.gradient!(first_deriv_compiled_tape, (yt[:,i], b[:,i], p...))[1]
            end
            y12_1_derivatives = hcat(y1_derivatives...)
            y1_1_deriv = y12_1_derivatives[1,:]
            y2_1_deriv = y12_1_derivatives[2,:]


            y1_2_deriv = map(1:batch_size) do i
                return ReverseDiff.gradient!(second_deriv_compiled_tape, (yt[:,i], b[:,i], p...))[1][1]
            end
        # elseif PI_use_AD
        #     ϵ = Float64(sqrt(eps(Float32)))
        #     preds_p_ϵ0 = eval_trunk_and_combine(yt .+ [ϵ,0],b,p)
        #     preds_m_ϵ0 = eval_trunk_and_combine(yt .- [ϵ,0],b,p)
        #     preds_p_0ϵ = eval_trunk_and_combine(yt .+ [0,ϵ],b,p)
        #     preds_m_0ϵ = eval_trunk_and_combine(yt .- [0,ϵ],b,p)
        #
        #     y1_2_deriv = (preds_p_ϵ0 .+ preds_m_ϵ0 .- 2 * preds)/ϵ^2
        #     y1_1_deriv = (preds_p_ϵ0 .- preds_m_ϵ0)/(2*ϵ)
        #     y2_1_deriv = (preds_p_0ϵ .- preds_m_0ϵ)/(2*ϵ)

        else
            preds_p_ϵ0 = eval_trunk_and_combine(yt .+ [ϵ,0],b,p)
            preds_m_ϵ0 = eval_trunk_and_combine(yt .- [ϵ,0],b,p)
            preds_p_0ϵ = eval_trunk_and_combine(yt .+ [0,ϵ],b,p)
            preds_m_0ϵ = eval_trunk_and_combine(yt .- [0,ϵ],b,p)

            y1_2_deriv = (preds_p_ϵ0 .+ preds_m_ϵ0 .- 2 * preds)/ϵ^2
            y1_1_deriv = (preds_p_ϵ0 .- preds_m_ϵ0)/(2*ϵ)
            y2_1_deriv = (preds_p_0ϵ .- preds_m_0ϵ)/(2*ϵ)
        end
        # y1_2_deriv = ReverseDiff.jacobian(first_deriv, (yt, b))[1]




        # y1_2_deriv = ReverseDiff.gradient(dy1->ReverseDiff.gradient(y1 -> combine_latent(model,evaluate_trunk(model,[y1' ; yt[2,i]]),b[:,i])[], dy1), [yt[1,i]])[]
        physics_loss_interior = sum((D * y1_2_deriv .- vel * y1_1_deriv .- y2_1_deriv).^2)
        # println(physics_loss_interior)


        #Boundary
        # physics_loss_squared += sum((model([yspan[1,1] * ones(1,size(yt,2)) ; yt[2,:]'], u_vals) .- model([yspan[1,2] * ones(1,size(yt,2)) ; yt[2,:]'], u_vals)).^2)
        # physics_loss_squared += sum(cat_preds[:,2*batch_size+1:3*batch_size].^2)  # because inner product is linear operation
        physics_loss_boundary = sum((combine_latent(model,t_left-t_right,b,p)).^2)  # because inner product is linear operation
        # println("")
        # println(physics_loss_squared)

        #Initial
        # initial_values = [interpolate(
        #     (x_locs_full,),
        #     cat(u_vals[:,i],u_vals[1,i],dims=1),
        #     Gridded(Interpolations.Linear())
        # )(yt[1,i]) for i=1:size(yt,2)]
        # physics_loss_squared += sum((cat_preds[:,1*batch_size+1:2*batch_size] .- random_sensors).^2)
        physics_loss_initial = sum((combine_latent(model,t_sensors,b,p) .- random_sensors).^2)
        # println(physics_loss_squared)

        # Data
        # data_loss_squared = sum((cat_preds[:,0*batch_size+1:1*batch_size] .- v_y_true).^2)
        data_loss_squared = sum((preds .- v_y_true).^2)

        # println(data_loss_squared)

        if p==nothing
            regularisation_loss = sum(norm(Flux.params(model)))
        else
            regularisation_loss = sum(norm(p))
        end
        return 2*(data_loss_squared * data_weight + physics_loss_initial * physics_weight_initial + physics_loss_boundary * physics_weight_boundary + physics_loss_interior * physics_weight_interior) / batch_size + regularisation_loss * regularisation_weight
    end
else
    loss((y, u_vals), v_y_true) = Flux.mse(model(y,u_vals), v_y_true)
end

if PI_use_AD
    @time loss(first(loaders.train)[1]..., params)
    first_deriv_compiled_tape = nothing
    second_deriv_compiled_tape = nothing
else
    @time loss(first(loaders.train)[1]...)
end
flush(stdout)

# d=first(loaders.train)[1]
# println(PI_use_AD ? loss(d..., params) : loss(d...))

## Training loop
if PI_use_AD
    opt = Optimisers.NAdam()
else
    opt = Flux.NAdam()
end
# opt = Adam()


loss_train = fill(NaN,n_epochs)
loss_validation = fill(NaN,n_epochs)
verbose = 0
train!(model, loaders, params, loss, opt, n_epochs, loss_train, loss_validation, verbose)

# To be used only after final model is selected
function compute_total_loss(loader)
    loss_test = 0
    for (d,s) in loader
        loss_test+=loss(d...)/length(loader)
    end
    return loss_test
end
loss_test = compute_total_loss(loaders.test)
# println(@sprintf "Test loss: %.3e" loss_test)


flush(stdout)
print("Mean of last 10 validation errors:\n$(mean(loss_validation[end-10:end]))")


## Plotting

if do_plots
    file_time_label = training_var_time ? "var" : "fixed"

    if training_var_time
        plot_seed = n_u_trajectories + n_u_trajectories_validation + n_u_trajectories_test÷3+11
        t_plot = range(start=ti, stop=tf, length=100)
        yt = hcat([[x,t] for x=x_locs_full for t=t_plot]...)
        u_vals_plot = u_func(x_locs_full, plot_seed)
        v_vals_plot = reshape(v_func(yt, plot_seed), length(t_plot), length(x_locs_full))
        deepo_solution = reshape(model(yt, u_vals_plot[begin:end-1])[:], length(t_plot), length(x_locs_full))
        # title = @sprintf "Example DeepONet input/output. MSE %.2e" Flux.mse(deepo_solution, v_vals_plot)
        p1=heatmap(x_locs_full, t_plot, deepo_solution, reuse = false, title="DeepONet\nprediction", clim=extrema([v_vals_plot;deepo_solution]),xticks=[ti,(ti+tf)/2,tf])
        xlabel!("y")
        ylabel!("t")
        title=@sprintf "Error\nMSE %.2e" Flux.mse(deepo_solution, v_vals_plot)
        p2=heatmap(x_locs_full, t_plot, v_vals_plot-deepo_solution, reuse = false, title=title, yticks=false,xticks=[ti,(ti+tf)/2,tf])
        xlabel!("y")
        p3=heatmap(x_locs_full, t_plot, v_vals_plot, reuse = false, title="Numerical\nsolution", clim=extrema([v_vals_plot;deepo_solution]), yticks=false,xticks=[ti,(ti+tf)/2,tf])
        xlabel!("y")
        p = plot(p1, p2, p3, reuse = false, layout = (1,3))
        savefig(p, "plots/convection_diffusion_example_$(file_time_label)_time.pdf")
        display(p)
    else
        plot_seed = n_u_trajectories + n_u_trajectories_validation + n_u_trajectories_test÷2
        x_locs_plot = cat(x_locs,xf,dims=1)
        u_vals_plot = u_func(x_locs_plot, plot_seed)
        v_vals_plot = v_func([x_locs_plot';tf*ones(1,size(x_locs_plot)...)], plot_seed)
        deepo_solution = model(reshape(x_locs_plot,1,:), u_vals_plot[begin:end-1])[:]
        title = @sprintf "Example DeepONet input/output. MSE %.2e" Flux.mse(deepo_solution, v_vals_plot)
        p=plot(x_locs_plot, u_vals_plot, label="Input function from test set", reuse = false, title=title, legend_position=:bottomright)
        plot!(x_locs_plot, v_vals_plot, label="Numerical solution")
        plot!(x_locs_plot, deepo_solution, label="DeepONet output")
        xlabel!("y")
        ylabel!("Function value")
        savefig(p, "plots/convection_diffusion_example_$(file_time_label)_time.pdf")
        display(p)
    end

    if do_plots && D==0
        FD_v_vals_plot = reshape(v_func_FD(yt, plot_seed), length(t_plot), length(x_locs_full))
        analytical_solution = hcat([u_func(x_locs_full .- vel*t, plot_seed) for t in t_plot]...)'
        p1=heatmap(x_locs_full, t_plot, analytical_solution, reuse = false, title="Analytical", clim=extrema([analytical_solution;FD_v_vals_plot;v_vals_plot]),xticks=[ti,(ti+tf)/2,tf])
        xlabel!("y")
        ylabel!("t")
        tit = @sprintf "FD numerical\nMSE: %.2e" mean((analytical_solution-FD_v_vals_plot).^2)
        p2=heatmap(x_locs_full, t_plot, FD_v_vals_plot, reuse = false, title=tit, clim=extrema([analytical_solution;FD_v_vals_plot;v_vals_plot]),xticks=[ti,(ti+tf)/2,tf])
        xlabel!("y")
        tit = @sprintf "fft numerical\nMSE: %.2e" mean((analytical_solution-v_vals_plot).^2)
        p3=heatmap(x_locs_full, t_plot, v_vals_plot, reuse = false, title=tit, clim=extrema([analytical_solution;FD_v_vals_plot;v_vals_plot]),xticks=[ti,(ti+tf)/2,tf])
        xlabel!("y")
        p = plot(p1, p2, p3, reuse = false, layout = (1,3))
        savefig(p, "plots/convection_example_$(file_time_label)_time.pdf")
        display(p)
    end

    p=plot(loss_train, label="Train", legend=:topright, reuse = false, markershape = :circle, yaxis=:log, title="DeepONet training progress")
    plot!(loss_validation, label="Validation", markershape = :circle)
    xlabel!("Epochs")
    ylabel!("Loss (MSE)")
    savefig(p, "plots/convection_diffusion_training_$(file_time_label)_time.pdf")
    display(p)
end
