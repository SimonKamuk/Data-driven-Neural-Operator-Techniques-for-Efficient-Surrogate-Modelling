using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, GraphNeuralNetworks, Statistics, LinearAlgebra, Graphs, SparseArrays, Juno, ForwardDiff, NeuralOperators

tspan = [0, 1]
s0 = 0
function f(s, p, t)
    ds = p[1](t, p[2])
    return ds
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
xi = tspan[1]
xf = tspan[2]
recompute_data = false

grf_generate_point_locs = range(start=xi, stop=xf, length=n_grf_generate_points)
sensor_locs = range(start=xi, stop=xf, length=n_sensors)
y_locs = range(start=tspan[1], stop=tspan[2], length=n_y_eval)



if !(@isdefined grf) | recompute_data
    kernel = Gaussian(l, σ=1, p=2)
    cov = CovarianceFunction(n_dims, kernel)
    grf = GaussianRandomField(cov, Spectral(), grf_generate_point_locs)
end

function u(x, seed)
    # Sensor vals
    interp = interpolate(
        (grf_generate_point_locs,),
        sample(grf,xi=randn(MersenneTwister(seed), randdim(grf))),
        Gridded(Interpolations.Linear()))
    return interp(x)
end

function s_numeric(y, seed)
    prob = ODEProblem(f, s0, tspan, [u, seed])
    sol = solve(prob, RK4(), saveat=y)
    return sol.u
end





## Generate training data
if !(@isdefined train_loader) | recompute_data
    sensor_vals = zeros((n_sensors, n_u_trajectories+n_u_trajectories_test, n_y_eval))
    s_vals = zeros((n_u_trajectories+n_u_trajectories_test, n_y_eval))
    seeds = zeros(Int, (n_u_trajectories + n_u_trajectories_test, n_y_eval))
    for seed in 1:n_u_trajectories+n_u_trajectories_test
        sensor_vals[:, seed, :] .= u(collect(sensor_locs), seed)
        s_vals[seed, :] = s_numeric(y_locs, seed)
        seeds[seed, :] .= seed
    end


    y_idx = repeat(reshape(Array(1:n_y_eval), 1, n_y_eval), n_u_trajectories+n_u_trajectories_test, 1)
    sensor_vals = reshape(permutedims(sensor_vals,[1,3,2]), n_sensors, :)
    s_vals = reshape(permutedims(s_vals,[2,1]), 1, :)
    y_idx = reshape(permutedims(y_idx,[2,1]), 1, :)
    seeds = reshape(permutedims(seeds,[2,1]), :)

end

data = ((y_locs[y_idx], sensor_vals), s_vals)
train_data, test_data = Flux.splitobs(data, at=n_u_trajectories*n_y_eval, shuffle=false)
train_seeds, test_seeds = Flux.splitobs(seeds, at=n_u_trajectories*n_y_eval, shuffle=false)

train_loader = Flux.DataLoader(train_data, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
test_loader = Flux.DataLoader(test_data, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
train_seeds_loader = Flux.DataLoader(train_seeds, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
test_seeds_loader = Flux.DataLoader(test_seeds, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)

first_x_test,first_y_test = first(test_loader)
first_seeds_test = first(test_seeds_loader)
y_locstest, sensor_valstest = first_x_test
for i in [1,2,3,batch_size ÷ 2,batch_size-3, batch_size-2,batch_size-1,batch_size]
    @assert s_numeric(y_locstest[:, i], first_seeds_test[i])[] == first_y_test[i]
    @assert u(sensor_locs, first_seeds_test[i]) == sensor_valstest[:,i]
end

first_x_train,first_y_train = first(train_loader)
first_seeds_train = first(train_seeds_loader)
y_locstrain, sensor_valstrain = first_x_train
for i in [1,2,3,batch_size ÷ 2,batch_size-3, batch_size-2,batch_size-1,batch_size]
    @assert s_numeric(y_locstrain[:, i], first_seeds_train[i])[] == first_y_train[i]
    @assert u(sensor_locs, first_seeds_train[i]) == sensor_valstrain[:,i]
end


## Prepare graphs
cutoff = 0.1
g_matrix = zeros(n_sensors,n_sensors)
for (idx, sensor) in enumerate(sensor_locs)
    g_matrix[idx, :] .= (xi-xf).^2 ./ (sensor .- sensor_locs).^2
end
g_matrix[diagind(g_matrix)] .= 0
g_matrix /= maximum(g_matrix)
g_matrix[g_matrix .< cutoff] .= 0
g_matrix[diagind(g_matrix)] .= 1
graph = GNNGraph(sparse(g_matrix), graph_type=:sparse)
batch_graphs = Flux.batch([GNNGraph(graph, graph_type=:coo) for _ in 1:batch_size])



## Define layers

use_gnn = true
if use_gnn
    # Ret god. Men langsom
    graph_conv_features = 5
    graph_conv_features_out = 5
    branch = GNNChain(
        GCNConv(1 => graph_conv_features, activation_function; use_edge_weight=true, add_self_loops=false),
        GCNConv(graph_conv_features => graph_conv_features_out, activation_function; use_edge_weight=true, add_self_loops=false),
        x->reshape(x,n_sensors*graph_conv_features_out,:),
        Dense(n_sensors*graph_conv_features_out, latent_size)
    )

    # graph_conv_features = 5
    # branch = GNNChain(
    #     GCNConv(1 => graph_conv_features, activation_function; use_edge_weight=true, add_self_loops=false),
    #     GCNConv(graph_conv_features => latent_size, activation_function; use_edge_weight=true, add_self_loops=false),
    #     GlobalPool(mean),
    #     x->reshape(x,latent_size,:),
    # )

    # branch = GNNChain(
    #     GCNConv(1 => nn_width, activation_function; use_edge_weight=true, add_self_loops=false),
    #     GlobalPool(mean),
    #     Dense(nn_width, latent_size)
    # )


    # Ikke så god
    # graph_conv_features = 5
    # branch = GNNChain(
    #     GCNConv(1 => graph_conv_features, activation_function; use_edge_weight=true, add_self_loops=false),
    #     GCNConv(graph_conv_features => nn_width, activation_function; use_edge_weight=true, add_self_loops=false),
    #     GlobalPool(mean),
    #     Dense(nn_width, nn_width),
    #     Dense(nn_width, latent_size)
    # )

else

    branch = Chain(
        Dense(n_sensors, nn_width, activation_function),
        Dense(nn_width, latent_size)
    )

end

trunk = Chain(
    Dense(1, nn_width, activation_function),
    Dense(nn_width, nn_width, activation_function),
    Dense(nn_width, latent_size, activation_function)
)
b0 = [0.0]


## Define model
struct NeuralOperator{branch_type}
    branch::branch_type
    trunk::Chain
    b0::Vector{Float64}
end


function (m::NeuralOperator{T})(y,sensor_vals) where T<:GNNChain

    this_batch_size = size(sensor_vals,2)
    if this_batch_size == batch_size
        return sum(m.branch(batch_graphs, reshape(sensor_vals,1,:)) .* m.trunk(y), dims=1) .+ m.b0[1]
    else
        this_batch_graphs = Flux.batch([GNNGraph(graph, graph_type=:coo) for _ in 1:this_batch_size])
        return sum(m.branch(this_batch_graphs, reshape(sensor_vals,1,:)) .* m.trunk(y), dims=1) .+ m.b0[1]
    end

end


function (m::NeuralOperator{T})(y,sensor_vals) where T<:Chain
    return sum(m.branch(sensor_vals) .* m.trunk(y), dims=1) .+ m.b0[1]
end





Flux.@functor NeuralOperator

neural_op = NeuralOperator{typeof(branch)}(branch, trunk, b0)

loss((y, sensor_vals), s_y_true) = Flux.mse(neural_op(y,sensor_vals), s_y_true)
params = Flux.params(neural_op)

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
prob = ODEProblem(f, s0, tspan, [u, plot_seed])
sol = solve(prob, RK4(), saveat=sensor_locs)
sensor_vals_plot = reshape(u(sensor_locs, plot_seed),:,1)
deepo_solution = neural_op(reshape(sol.t,1,:), sensor_vals_plot)[:]
deepo_deriv = ForwardDiff.derivative.(t->neural_op([t], sensor_vals_plot)[], sensor_locs)

p1=plot(sol.t, t->u(t,plot_seed), label="Input function", reuse = false)
plot!(sol.t, sol.u, label="Numerical solution")
plot!(sol.t, deepo_solution, label="DeepONet")
plot!(sol.t, deepo_deriv, label="DeepONet derivative")
display(p1)




#
# println("DeepONet solution: $(Flux.mse(sol.u,deepo_solution))")
# for i in 1:10
#     global deepo_solution
#     deepo_solution -= neural_op(reshape(sol.t,1,:), u(sensor_locs, plot_seed)-ForwardDiff.derivative.(t->neural_op([t], deepo_solution)[], sensor_locs))[:]
#     println("DeepONet iterative solution $i: $(Flux.mse(sol.u,deepo_solution))")
# end




p2=plot(loss_train, label="Train loss", legend=:topright, reuse = false, markershape = :circle)
plot!(loss_test, label="Test loss", markershape = :circle)
display(p2)


println(loss_test[end-3:end])
