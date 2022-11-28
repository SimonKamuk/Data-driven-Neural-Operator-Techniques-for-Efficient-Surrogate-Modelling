using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, GraphNeuralNetworks, Statistics, LinearAlgebra, Graphs, SparseArrays, Juno, ForwardDiff, Printf, ProgressBars, Distributions, FFTW

include("MyDeepONet.jl")
using .MyDeepONet

tspan = [0, 5]
yspan = [0, 1]
n_dims = 1
n_sensors = 100
nn_width = 70
latent_size = 70
activation_function = relu
n_u_trajectories = 1000
n_u_trajectories_test = 1000
n_u_trajectories_validation = 1000
batch_size = 50
n_spatial_finite_diff = n_sensors * 100
n_fft_diff = 1000
xi = yspan[1]
xf = yspan[2]
recompute_data = true
Random.seed!(0)
flux_ini = Flux.glorot_uniform(MersenneTwister(rand(Int64)))

n_y_locs = 1000
y_locs = range(start=xi, stop=xf, length=n_y_locs+1)[begin:end-1]


D=200
vel=10



## For the finite difference, method of lines
spatial_finite_diff_locs = range(start=xi, stop=xf, length=n_spatial_finite_diff+1)[begin:end-1]
# If we did not use periodic boundary, then I would need x_locs to start and end on the boundaries.
@assert (n_spatial_finite_diff/n_sensors) % 1.0 <= 1e-8 # Check that it is close to integer, so sensor points and FD points line up
finite_diff_to_x_locs_idx = (0:n_sensors-1)*round(Int,n_spatial_finite_diff/n_sensors) .+ 1
x_locs_FD = spatial_finite_diff_locs[finite_diff_to_x_locs_idx] # Sensor locations (input function evaluation points)



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
        -2*D*c[i] + (D-vel/2)*c[ip1] + (D+vel/2)*c[im1]
    end
    for i in 1:n]

    # above is roughly twice as fast as below with A defined as in comment above
    #dc = A * c

    return dc
end





## For solving the pde as coupled odes in frequency domain
base_wave_number = 2*π/(yspan[end]-yspan[begin])
wave_numbers = (1:n_fft_diff÷2).*base_wave_number

fft_diff_locs = range(start=xi, stop=xf, length=n_fft_diff÷2+1)[begin:end-1]
x_locs = range(start=xi, stop=xf, length=n_sensors+1)[begin:end-1]

omega = fftfreq(n_fft_diff÷2)
coef_vec = -D*omega.^2 .- vel*omega*im
function f_fft(fft_c, p, t)
    return coef_vec .* fft_c
end


## Define functions


function get_u(seed)
    # Define input function
    rng = MersenneTwister(seed)
    A = rand(rng, Uniform(0, 10), n_fft_diff÷2) .* sqrt.(exp.(-wave_numbers.^2 ./ (2*base_wave_number^2)))
    ϕ = rand(rng, Uniform(0, 2*π), n_fft_diff÷2)

    return x->sum([A[i]*cos.(wave_numbers[i]*x.+ϕ[i]) for i in 1:n_fft_diff÷2])
end


# get_u(seed) = x->1/(0.1 * sqrt(2*π)) * exp.(-1/2 * ((x.-0.5)/0.1).^2)

function v_func_FD(y, seed; manual_u = nothing)
    # Solve problem with numerical ode solver (4th order Runge-Kutta) and
    # evaluate solution at points y

    if manual_u == nothing
        u = get_u(seed)
        prob = ODEProblem(f_FD, u(spatial_finite_diff_locs), tspan, saveat=[tspan[end]])
    else
        prob = ODEProblem(f_FD, manual_u, tspan, saveat=[tspan[end]])
    end
    v_values = solve(prob, Tsit5()).u[end]

    inter = interpolate(
        (cat(spatial_finite_diff_locs,xf, dims=1),),
        cat(v_values,v_values[begin],dims=1),
        Gridded(Interpolations.Linear()))

    return inter(y)
end

function v_func_fft(y, seed; manual_u = nothing)
    # Solve problem with numerical ode solver (4th order Runge-Kutta) and
    # evaluate solution at points y

    if manual_u == nothing
        u = get_u(seed)
        u_fft = fft(u(fft_diff_locs))
    else
        u_fft = fft(manual_u)
    end
    prob = ODEProblem(f_fft, u_fft, tspan, saveat=[tspan[end]])

    v_values = real(ifft(solve(prob, Tsit5()).u[end]))

    # inter = interpolate(
    #     (fft_diff_locs,),
    #     v_values,
    #     Gridded(Interpolations.Linear()))

    inter = interpolate(
        (cat(fft_diff_locs,xf, dims=1),),
        cat(v_values,v_values[begin],dims=1),
        Gridded(Interpolations.Linear()))

    return inter(y)
end


v_func = v_func_fft

u_func(x_locs, seed) = get_u(seed)(x_locs)


## Generate data
if !(@isdefined loaders) | recompute_data
    loaders = generate_data(x_locs, yspan, u_func, v_func, n_sensors, n_u_trajectories, n_u_trajectories_validation, n_u_trajectories_test, n_sensors, batch_size, equidistant_y=false)
    # loaders = generate_data(x_locs, yspan, u_func, v_func, n_sensors, n_u_trajectories, n_u_trajectories_validation, n_u_trajectories_test, n_sensors, batch_size, y_locs=x_locs)

    # code_lowered(get_u)[]
end



## Define layers

branch = Chain(
    Dense(n_sensors, nn_width, activation_function, init=flux_ini),
    Dense(nn_width, nn_width, activation_function, init=flux_ini),
    Dense(nn_width, nn_width, activation_function, init=flux_ini),
    Dense(nn_width, latent_size, init=flux_ini)
)
trunk = Chain(
    Dense(1, nn_width, activation_function, init=flux_ini),
    Dense(nn_width, nn_width, activation_function, init=flux_ini),
    Dense(nn_width, nn_width, activation_function, init=flux_ini),
    Dense(nn_width, nn_width, activation_function, init=flux_ini),
    Dense(nn_width, latent_size, activation_function, init=flux_ini)
)

# Define model
model = DeepONet(trunk=trunk, branch=branch, const_bias=true)
loss((y, u_vals), v_y_true) = Flux.mse(model(y,u_vals), v_y_true)
params = Flux.params(model)


loss(first(loaders.train)...)


## Training loop
opt = NAdam()
# opt = Adam()

n_epochs = 100

loss_train, loss_validation = train!(loaders, params, loss, opt, n_epochs)

# To be used only after final model is selected
function get_loss_test()
    loss_test = 0
    for d in loaders.test
        loss_test+=loss(d...)/length(loaders.test)
    end
    return loss_test
end
# loss_test = get_loss_test()
# println(@sprintf "Test loss: %.3e" loss_test)






## Plotting
plot_seed = n_u_trajectories + n_u_trajectories_validation + n_u_trajectories_test÷2
u_vals_plot = u_func(x_locs, plot_seed)
v_vals_plot = v_func(x_locs, plot_seed)
deepo_solution = model(reshape(x_locs,1,:), u_vals_plot)[:]
title = @sprintf "Example DeepONet input/output. MSE %.2e" Flux.mse(deepo_solution, v_vals_plot)
p=plot(x_locs, u_vals_plot, label="Input function from test set", reuse = false, title=title)
plot!(x_locs, v_vals_plot, label="Numerical solution")
plot!(x_locs, deepo_solution, label="DeepONet output")
xlabel!("y")
ylabel!("Function value")
savefig(p, "plots/convection_diffusion_test_function.pdf")
display(p)





p=plot(loss_train, label="Train", legend=:topright, reuse = false, markershape = :circle, yaxis=:log, title="DeepONet training progress")
plot!(loss_validation, label="Validation", markershape = :circle)
xlabel!("Epochs")
ylabel!("Loss (MSE)")
savefig(p, "plots/convection_diffusion_training.pdf")
display(p)
