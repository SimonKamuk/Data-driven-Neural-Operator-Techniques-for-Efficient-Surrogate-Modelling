using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, GraphNeuralNetworks, Statistics, LinearAlgebra, Graphs, SparseArrays, Juno, ForwardDiff, Printf, ProgressBars


include("MyDeepONet.jl")
using .MyDeepONet


yspan = [0, 1]
v0 = 0 # initial value of solution at y=0
# Define problem to be solved: dv/dy = u(y)
function f(v, u, y)
    dv = u(y)
    return dv
end

l = 0.2 * sqrt(2)
n_dims = 1
n_sensors = 100
nn_width = 50
latent_size = 50 # p
activation_function = relu
n_grf_generate_points = 1000
n_u_trajectories = 100
n_u_trajectories_test = 30
n_u_trajectories_validation = 30
batch_size = 80
n_y_eval = 1000
xi = yspan[1]
xf = yspan[2]
recompute_data = false


x_locs = range(start=xi, stop=xf, length=n_sensors) # Sensor locations (input function evaluation points)
y_locs = range(start=yspan[1], stop=yspan[2], length=n_y_eval) # Output function evaluation points



## Define functions
if !(@isdefined grf) | recompute_data
    println("Generating gaussian random field")
    kernel = Gaussian(l, σ=1, p=2)
    cov = CovarianceFunction(n_dims, kernel)
    grf_generate_point_locs = range(start=xi, stop=xf, length=n_grf_generate_points)
    grf = GaussianRandomField(cov, Spectral(), grf_generate_point_locs)
end

function get_u(seed)
    # Define input function
    interp = interpolate(
        (grf_generate_point_locs,),
        sample(grf,xi=randn(MersenneTwister(seed), randdim(grf))),
        Gridded(Interpolations.Linear()))
    return interp
end

function v_func(y, seed)
    # Solve problem with numerical ode solver (4th order Runge-Kutta) and
    # evaluate solution at points y
    prob = ODEProblem(f, v0, yspan, get_u(seed))
    v_values = solve(prob, RK4(), saveat=y).u
    return v_values
end

u_func(x_locs, seed) = get_u(seed)(x_locs)


## Generate data
if !(@isdefined loaders) | recompute_data
    loaders = generate_data(x_locs, y_locs, u_func, v_func, n_sensors, n_u_trajectories, n_u_trajectories_test, n_u_trajectories_validation, n_y_eval, batch_size)
end

## Define layers
branch = Chain(
    Dense(n_sensors, nn_width, activation_function),
    Dense(nn_width, latent_size)
)
trunk = Chain(
    Dense(1, nn_width, activation_function),
    Dense(nn_width, nn_width, activation_function),
    Dense(nn_width, latent_size, activation_function)
)

# Define model
model = DeepONet(trunk=trunk, branch=branch, const_bias=true)
loss((y, u_vals), v_y_true) = Flux.mse(model(y,u_vals), v_y_true)
params = Flux.params(model)


## Training loop
opt = NAdam()
# opt = Adam()

n_epochs = 70

# loss_train, loss_validation = train!(loaders, params, loss, opt, n_epochs, patience=Inf,threshold_factor = 0.85, lr_factor = 0.5, cooldown=3)
loss_train, loss_validation = train!(loaders, params, loss, opt, n_epochs)


# To be used only after final model is selected
# loss_test = 0
# for d in loaders.test
#     loss_test+=loss(d...)/length(loaders.test)
# end
# println(@sprintf "Test loss: %.3e" loss_test)






## Plotting
plot_seed = 50
u = get_u(plot_seed)
prob = ODEProblem(f, v0, yspan, u)
sol = solve(prob, RK4(), saveat=x_locs)
u_vals_plot = reshape(u(x_locs),:,1)
deepo_solution = model(reshape(x_locs,1,:), u_vals_plot)[:]
deepo_deriv = ForwardDiff.derivative.(y->model([y], u_vals_plot)[], x_locs)

p1=plot(x_locs, x->u(x), label="Input function", reuse = false)
plot!(x_locs, sol.u, label="Numerical solution")
plot!(x_locs, deepo_solution, label="DeepONet")
plot!(x_locs, deepo_deriv, label="DeepONet derivative")
display(p1)



unseen_u_vals = sin.(x_locs)
unseen_v_vals = -cos.(y_locs)
unseen_deepo_vals = reshape(model(reshape(y_locs,1,:), unseen_u_vals),:)
unseen_deepo_deriv = ForwardDiff.derivative.(y->model([y], unseen_u_vals)[], x_locs)
p2 = plot(x_locs, unseen_u_vals, label="Input: sin", reuse = false)
plot!(y_locs, unseen_v_vals, label="Analytical output: -cos")
plot!(y_locs, unseen_deepo_vals, label="DeepONet output")
plot!(y_locs, unseen_deepo_vals .- 1, label="DeepONet output - 1")
plot!(x_locs, unseen_deepo_deriv, label="DeepONet output derivative")
display(p2)


unseen_u_vals = cos.(5.8π * x_locs)
unseen_v_vals = sin.(5.8π * y_locs) / (5.8π)
unseen_deepo_vals = reshape(model(reshape(y_locs,1,:), unseen_u_vals),:)
unseen_deepo_deriv = ForwardDiff.derivative.(y->model([y], unseen_u_vals)[], x_locs)
p3 = plot(x_locs, unseen_u_vals, label="Input: cos", reuse = false)
plot!(y_locs, unseen_v_vals, label="Analytical output: sin")
plot!(y_locs, unseen_deepo_vals, label="DeepONet output")
plot!(x_locs, unseen_deepo_deriv, label="DeepONet output derivative")
display(p3)





p4=plot(loss_train, label="Train loss", legend=:topright, reuse = false, markershape = :circle, yaxis=:log)
plot!(loss_validation, label="Validation loss", markershape = :circle)
display(p4)
