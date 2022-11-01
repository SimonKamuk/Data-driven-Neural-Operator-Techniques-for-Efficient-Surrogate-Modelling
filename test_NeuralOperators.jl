using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, GraphNeuralNetworks, Statistics, LinearAlgebra, Graphs, SparseArrays, Juno, ForwardDiff, NeuralOperators
import Base.@kwdef

yspan = [0, 1]
v0 = 0 # initial value of solution at y=0
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
batch_size = 80
n_y_eval = 1000
xi = yspan[1]
xf = yspan[2]
recompute_data = false


x_locs = range(start=xi, stop=xf, length=n_sensors) # Sensor locations (input function evaluation points)
y_locs = range(start=yspan[1], stop=yspan[2], length=n_y_eval) # Output function evaluation points



if !(@isdefined grf) | recompute_data
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

function v_numerical(y, u)
    # Solve problem with numerical ode solver (4th order Runge-Kutta) and
    # evaluate solution at points y
    prob = ODEProblem(f, v0, yspan, u)
    v_values = solve(prob, RK4(), saveat=y).u
    return v_values
end





## Generate training data
if !(@isdefined train_loader) | recompute_data
    u_vals = zeros((n_sensors, n_u_trajectories+n_u_trajectories_test, n_y_eval))
    v_vals = zeros((n_u_trajectories+n_u_trajectories_test, n_y_eval))
    seeds = zeros(Int, (n_u_trajectories + n_u_trajectories_test, n_y_eval))
    for seed in 1:n_u_trajectories+n_u_trajectories_test
        u=get_u(seed)
        u_vals[:, seed, :] .= u(collect(x_locs))
        v_vals[seed, :] = v_numerical(y_locs, u)
        seeds[seed, :] .= seed
    end


    y_idx = repeat(reshape(Array(1:n_y_eval), 1, n_y_eval), n_u_trajectories+n_u_trajectories_test, 1)
    u_vals = reshape(permutedims(u_vals,[1,3,2]), n_sensors, :)
    v_vals = reshape(permutedims(v_vals,[2,1]), 1, :)
    y_idx = reshape(permutedims(y_idx,[2,1]), 1, :)
    seeds = reshape(permutedims(seeds,[2,1]), :)

end

data = ((y_locs[y_idx], u_vals), v_vals)
train_data, test_data = Flux.splitobs(data, at=n_u_trajectories*n_y_eval, shuffle=false)
train_seeds, test_seeds = Flux.splitobs(seeds, at=n_u_trajectories*n_y_eval, shuffle=false)

train_loader = Flux.DataLoader(train_data, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
test_loader = Flux.DataLoader(test_data, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
train_seeds_loader = Flux.DataLoader(train_seeds, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
test_seeds_loader = Flux.DataLoader(test_seeds, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)

first_input_test,first_output_test = first(test_loader)
first_seeds_test = first(test_seeds_loader)
y_locstest, u_valstest = first_input_test
for i in [1,2,3,batch_size ÷ 2,batch_size-3, batch_size-2,batch_size-1,batch_size]
    u = get_u(first_seeds_test[i])
    @assert v_numerical(y_locstest[:, i], u)[] == first_output_test[i]
    @assert u(x_locs) == u_valstest[:,i]
end

first_x_train,first_y_train = first(train_loader)
first_seeds_train = first(train_seeds_loader)
y_locstrain, u_valstrain = first_x_train
for i in [1,2,3,batch_size ÷ 2,batch_size-3, batch_size-2,batch_size-1,batch_size]
    u = get_u(first_seeds_train[i])
    @assert v_numerical(y_locstrain[:, i], u)[] == first_y_train[i]
    @assert u(x_locs) == u_valstrain[:,i]
end



## Define layers


b = [0.0]

branch = Chain(
    Dense(n_sensors, nn_width, activation_function),
    Dense(nn_width, latent_size),

)

trunk = Chain(
    Dense(1, nn_width, activation_function),
    Dense(nn_width, nn_width, activation_function),
    Dense(nn_width, latent_size, activation_function),

)

# branch
x->cat(x, ones(1,size(x,2)), dims=1)
# trunk
x->cat(x, fill(b[1],(1,size(x,2))), dims=1)


## Define model

# Option 1, hardcode bias into branch and trunks, make bias trainable
Flux.trainable(m::DeepONet) = (m.branch_net, m.trunk_net, b)
model = DeepONet(branch, trunk)

# Option 2, create struct to automatically append bias layers
@kwdef mutable struct DeepONet_with_bias
    branch::Any
    trunk::Any
    bias_behaviour::Symbol = :none

    
end


#=
arguments:
branch, trunk(, :none is implied)
branch, trunk, :constant
branch, trunk, :variable
=#

## Define loss
loss((y, u_vals), v_y_true) = Flux.mse(model(y,u_vals), v_y_true)
params = Flux.params(model)

loss(first(train_loader)...)
## Training loop
opt = ADAM(0.001)
n_epochs = 30

loss_train = zeros(n_epochs)
loss_test = zeros(n_epochs)
for e in 1:n_epochs
    println("Epoch: $e")
    Flux.train!(loss, params, train_loader, opt)
    for d in train_loader
        loss_train[e]+=loss(d...)/length(train_loader)
    end
    for d in test_loader
        loss_test[e]+=loss(d...)/length(test_loader)
    end
end


## Plotting
plot_seed = 50
u = get_u(plot_seed)
prob = ODEProblem(f, v0, yspan, u)
sol = solve(prob, RK4(), saveat=x_locs)
u_vals_plot = reshape(u(x_locs),:,1)
deepo_solution = neural_op(reshape(x_locs,1,:), u_vals_plot)[:]
deepo_deriv = ForwardDiff.derivative.(y->neural_op([y], u_vals_plot)[], x_locs)

p1=plot(x_locs, x->u(x), label="Input function", reuse = false)
plot!(x_locs, sol.u, label="Numerical solution")
plot!(x_locs, deepo_solution, label="DeepONet")
plot!(x_locs, deepo_deriv, label="DeepONet derivative")
display(p1)



unseen_u_vals = sin.(x_locs)
unseen_v_vals = -cos.(y_locs)
unseen_deepo_vals = reshape(neural_op(reshape(y_locs,1,:), unseen_u_vals),:)
unseen_deepo_deriv = ForwardDiff.derivative.(y->neural_op([y], unseen_u_vals)[], x_locs)
p2 = plot(x_locs, unseen_u_vals, label="Input: sin", reuse = false)
plot!(y_locs, unseen_v_vals, label="Analytical output: -cos")
plot!(y_locs, unseen_deepo_vals, label="DeepONet output")
plot!(x_locs, unseen_deepo_deriv, label="DeepONet output derivative")
display(p2)


unseen_u_vals = cos.(x_locs)
unseen_v_vals = sin.(y_locs)
unseen_deepo_vals = reshape(neural_op(reshape(y_locs,1,:), unseen_u_vals),:)
unseen_deepo_deriv = ForwardDiff.derivative.(y->neural_op([y], unseen_u_vals)[], x_locs)
p2 = plot(x_locs, unseen_u_vals, label="Input: cos", reuse = false)
plot!(y_locs, unseen_v_vals, label="Analytical output: sin")
plot!(y_locs, unseen_deepo_vals, label="DeepONet output")
plot!(x_locs, unseen_deepo_deriv, label="DeepONet output derivative")
display(p2)




unseen_v(y) = sin(10*y) * exp(2*y) + y + y^2 + 1/(y+0.1) - 10
expanded_y_locs = -0.05:0.001:2
unseen_v_vals = unseen_v.(expanded_y_locs)
unseen_u_vals = ForwardDiff.derivative.(unseen_v, x_locs)
unseen_u_vals_plot = ForwardDiff.derivative.(unseen_v, expanded_y_locs)
unseen_deepo_vals = reshape(neural_op(reshape(expanded_y_locs,1,:), unseen_u_vals),:)
unseen_deepo_deriv = ForwardDiff.derivative.(y->neural_op([y], unseen_u_vals)[], expanded_y_locs)
p2 = plot(expanded_y_locs, unseen_u_vals_plot, label="Input: 2y - 1/(y + 0.1)² + 2 exp(2y) sin(10 y) + 10 exp(2y) cos(10y) + 1", reuse = false, ylims=(-100,60), legend=:bottomleft)
plot!(expanded_y_locs, unseen_v_vals, label="Analytical output: sin(10y) exp(2y) + y + y² + 1/(y+0.1) - 10")
plot!(expanded_y_locs, unseen_deepo_vals, label="DeepONet output")
plot!(expanded_y_locs, unseen_deepo_deriv, label="DeepONet output derivative")
display(p2)


#
# println("DeepONet solution: $(Flux.mse(sol.u,deepo_solution))")
# for i in 1:10
#     global deepo_solution
#     deepo_solution -= neural_op(reshape(x_locs,1,:), u(x_locs, plot_seed)-ForwardDiff.derivative.(t->neural_op([t], deepo_solution)[], x_locs))[:]
#     println("DeepONet iterative solution $i: $(Flux.mse(sol.u,deepo_solution))")
# end




p3=plot(loss_train, label="Train loss", legend=:topright, reuse = false, markershape = :circle)
plot!(loss_test, label="Test loss", markershape = :circle)
display(p3)


println(loss_test[end-3:end])
