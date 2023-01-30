using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, GraphNeuralNetworks, Statistics, LinearAlgebra, Graphs, SparseArrays, Juno, ForwardDiff, Printf, ProgressBars

include("MyDeepONet.jl")
using .MyDeepONet

## Define hyperparameters for data generation and DeepONet architecture

l = 0.2 * sqrt(2) # GRF correlation length
n_sensors = 100 # Number of input function evaluations fed to branch net
nn_width = 50 # Width of both trunk and branch
activation_function = relu # activation in trunk net. Branch has no activation.
latent_size = 50 # p, number of outputs from each subnetwork
n_grf_generate_points = 1000 # Number of points generated from GRF to use for interpolating input functions
n_u_trajectories = 100 # Number of input functions, training
n_u_trajectories_test = 1000 # Number of input functions, testing
n_u_trajectories_validation = 100 # Number of input functions, validation
batch_size = 100
n_y_eval = 1000 # Number of output function evaluations per input function
yspan = [0 1] # Domain
xi = yspan[1]
xf = yspan[2]


recompute_data = true
Random.seed!(0)
flux_ini = Flux.glorot_uniform(MersenneTwister(rand(Int64)))
x_locs = range(start=xi, stop=xf, length=n_sensors) # Sensor locations (input function evaluation points)
y_locs = range(start=yspan[1], stop=yspan[2], length=n_y_eval) # Output function evaluation points


## Define input and output functions
if !(@isdefined grf) | recompute_data
    println("Generating gaussian random field")
    kernel = Gaussian(l, σ=1, p=2)
    cov = CovarianceFunction(1, kernel)
    grf_generate_point_locs = range(start=xi, stop=xf, length=n_grf_generate_points)
    grf = GaussianRandomField(cov, Spectral(), grf_generate_point_locs)
end

function get_u(seed)
    # Define input function based on gaussian random field
    interp = interpolate(
        (grf_generate_point_locs,),
        GaussianRandomFields.sample(grf,xi=randn(MersenneTwister(seed), randdim(grf))),
        Gridded(Interpolations.Linear()))
    return interp
end

v0 = 0 # initial value of solution at y=0
# Define problem to be solved: dv/dy = u(y)
function f(v, u, y)
    dv = u(y)
    return dv
end
function v_func(y, seed)
    # Solve problem with numerical ode solver
    # (4th order Runge-Kutta implemented in the Tsit5 algorithm).
    # evaluate solution at points y
    sort_idx = sortperm(y[:])
    prob = ODEProblem(f, v0, yspan, get_u(seed))
    v_values = solve(prob, Tsit5(), saveat=y).u
    return v_values[invperm(sort_idx)]
end

u_func(x_locs, seed) = get_u(seed)(x_locs)


## Generate data using generate_data function from MyDeepONet module
if !(@isdefined loaders) | recompute_data
    loaders = generate_data(x_locs, yspan, u_func, v_func, n_sensors, n_u_trajectories, n_u_trajectories_validation, n_u_trajectories_test, n_y_eval, batch_size; equidistant_y=false)
end


## Define layers
branch = Chain(
    Dense(n_sensors, latent_size, init=flux_ini)
)
trunk = Chain(
    Dense(1, nn_width, activation_function, init=flux_ini),
    Dense(nn_width, latent_size, activation_function, init=flux_ini)
)

# Define model using DeepONet struct from MyDeepONet module
model = DeepONet(trunk=trunk, branch=branch, const_bias_trainable=true)
loss(((y, u_vals), v_y_true),s) = Flux.mse(model(y,u_vals), v_y_true)
params = Flux.params(model)


## Prediction time and null guess
println("Evaluation times per batch:")
args = first(loaders.train)
@time model(args[1][1]...)
@time model(args[1][1]...)

all_v_vec::Vector{Float64} = []
for (d,s) in loaders.test
    append!(all_v_vec, d[2])
end
loss_null_guess=Flux.mse(zeros(size(all_v_vec)...), all_v_vec)
null_guess_string = @sprintf "Null guess, test loss (pure data): %.3e" loss_null_guess
println("")
println(null_guess_string)
flush(stdout)


## Training loop using train! function from MyDeepONet module
opt = NAdam()

n_epochs = 100

# loss_train, loss_validation = train!(loaders, params, loss, opt, n_epochs, patience=Inf,threshold_factor = 0.85, lr_factor = 0.5, cooldown=3)
loss_train, loss_validation = train!(model, loaders, params, loss, opt, n_epochs)

# To be used only after final model is selected
function get_loss_test()
    loss_test = 0
    for (d,s) in loaders.test
        loss_test+=loss(d,s)/length(loaders.test)
    end
    return loss_test
end
loss_test = get_loss_test()
println(@sprintf "Test loss: %.3e" loss_test)


## Plotting
plot_seed = n_u_trajectories + n_u_trajectories_validation + n_u_trajectories_test÷2
u_vals_plot = u_func(x_locs, plot_seed)
v_vals_plot = v_func(x_locs, plot_seed)
deepo_solution = model(reshape(x_locs,1,:), u_vals_plot)[:]
deepo_deriv = ForwardDiff.derivative.(y->model([y], u_vals_plot)[], x_locs)
title = "Example DeepONet input/output"
p1=plot(x_locs, u_vals_plot, label="Input function from test set", reuse = false, title=title)
plot!(x_locs, v_vals_plot, label="Numerical solution")
plot!(x_locs, deepo_solution, label="DeepONet")
plot!(x_locs, deepo_deriv, label="DeepONet derivative")
xlabel!("y")
ylabel!("Function value")
savefig(p1, "plots/antiderivative_test_function.pdf")
display(p1)

title = @sprintf "Example DeepONet error. MSE %.2e" Flux.mse(deepo_solution, v_vals_plot)
p1=plot(x_locs, v_vals_plot-deepo_solution, reuse = false, title=title, legend=false)
xlabel!("y")
ylabel!("Error")
savefig(p1, "plots/antiderivative_test_function_error.pdf")
display(p1)

unseen_u_vals = sin.(x_locs)
unseen_v_vals = -cos.(y_locs)
unseen_deepo_vals = reshape(model(reshape(y_locs,1,:), unseen_u_vals),:)
unseen_deepo_deriv = ForwardDiff.derivative.(y->model([y], unseen_u_vals)[], x_locs)
title = "Example DeepONet input/output"
p2 = plot(x_locs, unseen_u_vals, label="Input: sin(y)", reuse = false, title=title)
plot!(y_locs, unseen_v_vals .+ 1, label="Analytical solution: -cos(y) + 1")
plot!(y_locs, unseen_deepo_vals, label="DeepONet output")
plot!(x_locs, unseen_deepo_deriv, label="DeepONet output derivative")
xlabel!("y")
ylabel!("Function value")
savefig(p2, "plots/antiderivative_sinusoidal.pdf")
display(p2)

title = @sprintf "Example DeepONet error. MSE %.2e" Flux.mse(unseen_deepo_vals, unseen_v_vals .+ 1)
p2 = plot(y_locs, unseen_v_vals .+ 1 - unseen_deepo_vals, reuse = false, title=title, legend=false)
xlabel!("y")
ylabel!("Error")
savefig(p2, "plots/antiderivative_sinusoidal_error.pdf")
display(p2)

unseen_u_vals = cos.(7π * x_locs)
unseen_v_vals = sin.(7π * y_locs) / (7π)
unseen_deepo_vals = reshape(model(reshape(y_locs,1,:), unseen_u_vals),:)
unseen_deepo_deriv = ForwardDiff.derivative.(y->model([y], unseen_u_vals)[], x_locs)
title = "Example DeepONet input/output"
p3 = plot(x_locs, unseen_u_vals, label="Input: cos(7πy)", reuse = false, title=title)
plot!(y_locs, unseen_v_vals, label="Analytical solution: sin(7πy)/(7π)")
plot!(y_locs, unseen_deepo_vals, label="DeepONet output")
plot!(x_locs, unseen_deepo_deriv, label="DeepONet output derivative")
xlabel!("y")
ylabel!("Function value")
savefig(p3, "plots/antiderivative_high_freq.pdf")
display(p3)

title = @sprintf "Example DeepONet error. MSE %.2e" Flux.mse(unseen_deepo_vals, unseen_v_vals)
p3 = plot(y_locs, unseen_v_vals-unseen_deepo_vals, reuse = false, title=title, legend=false)
xlabel!("y")
ylabel!("Error")
savefig(p3, "plots/antiderivative_high_freq_error.pdf")
display(p3)

p4=plot(loss_train, label="Train", legend=:topright, reuse = false, markershape = :circle, yaxis=:log, title="DeepONet training progress")
plot!(loss_validation, label="Validation", markershape = :circle)
xlabel!("Epochs")
ylabel!("Loss (MSE)")
savefig(p4, "plots/antiderivative_training.pdf")
display(p4)
