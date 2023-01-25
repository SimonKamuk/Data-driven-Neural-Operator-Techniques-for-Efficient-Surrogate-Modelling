using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, GraphNeuralNetworks, Statistics, LinearAlgebra, Graphs, SparseArrays, Juno, ForwardDiff, Printf, ProgressBars, Distributions, FFTW, ReverseDiff, FileIO, JLD2, Optimisers

include("MyDeepONet.jl")
using .MyDeepONet



# Data setup
L = 50 # base wave length [m]
h = 10 # water depth, [m]
ρ = 1025 # saltwater density [kg/m^3]
g = 9.82 # acceleration of gravity [m/s^2]
H_range = [0.01,0.5]
xi = 0
xf = L
zi = -h
zf = 0
yspan = [xi xf;zi zf]
n_u_trajectories = 1000
n_u_trajectories_test = 1000
n_u_trajectories_validation = 1000
n_y_eval = 100
batch_size = 50
n_epochs = 100
n_freq_gen = 100
frequency_decay = 0.1  # Larger number means faster decay, meaning fewer high frequency components
Random.seed!(0)
flux_ini = Flux.glorot_uniform(MersenneTwister(rand(Int64)))
recompute_data = false
save_on_recompute = false
const_bias_trainable = false
trunk_var_bias = true
equidistant_y = false

# Model setup
if length(ARGS) == 0
    n_sensors = 50
    branch_width = 50
    trunk_width = 75
    branch_depth = 3
    trunk_depth = 4
    latent_size = 75
    activation_function = softplus
    do_plots = true
    save_model = true
    load_model = true
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


    # (branch_width,trunk_width,branch_depth,trunk_depth) = [
    # (bw, tw, bd, td)
    # for bw in [35, 50, 65]
    # for tw in [35, 50, 65]
    # for bd in [3,4,5,6]
    # for td in [3,4,5,6]
    # ][jobindex]
    # physics_weight_initial = 0.0
    # physics_weight_boundary = 0.0
    # physics_weight_interior = 0.0
    # n_sensors = 50
    # latent_size = 75
    # activation_function = softplus
    # data_weight = 1.0
    # regularisation_weight = 0.0


    PI_use_AD = false  # AD not currently working
    do_plots = false
end



x_locs = range(start=xi, stop=xf, length=n_sensors+1)[begin:end-1]
x_locs_full = cat(x_locs,xf,dims=1)


function ϕ(x,z,t,H,δ)

    ϕ_vec = map(1:n_freq_gen) do i
        g = 9.82
        c = sqrt(g*L/(2*π*i) * tanh(2*π*h*i/L))
        T = L/(c*i)
        k = 2*π*i/L
        ω = 2*π/T

        return @. -H[i]*c/2 * cosh(k*(z+h)) / sinh(k*h) * sin(ω*t-k*x+δ[i])
    end

    return sum(ϕ_vec)
end
function p_excess(x,z,t,H,δ)

    p_vec = map(1:n_freq_gen) do i
        c = sqrt(g*L/(2*π*i) * tanh(2*π*h*i/L))
        T = L/(c*i)
        k = 2*π*i/L
        ω = 2*π/T

        return @. ρ*g*H[i]/2 * cosh(k*(z+h)) / sinh(k*h) * cos(ω*t-k*x+δ[i])
    end

    return sum(p_vec)
end
function uw(x,z,t,H,δ)

    uw_vec = map(1:n_freq_gen) do i
        g = 9.82
        c = sqrt(g*L/(2*π*i) * tanh(2*π*h*i/L))
        T = L/(c*i)
        k = 2*π*i/L
        ω = 2*π/T

        u = @. π * H[i] / T * cosh(k*(z+h)) / sinh(k*h) * cos(ω*t-k*x+δ[i])
        w = @. - π * H[i] / T * sinh(k*(z+h)) / sinh(k*h) * sin(ω*t-k*x+δ[i])
        return [u,w]
    end

    return sum(uw_vec)
end


function u_func(x, seed)
    # Define input function
    rng = MersenneTwister(seed)
    H = rand(rng, Uniform(H_range[1],H_range[2]), n_freq_gen) .* exp.(-frequency_decay *  (0:n_freq_gen-1).^2)
    δ = rand(rng, Uniform(0, 2*π), n_freq_gen)

    return ϕ(x,0,0,H,δ)
end
function v_func(y, seed)
    x = y[1,:]
    z = y[2,:]

    rng = MersenneTwister(seed)
    H = rand(rng, Uniform(H_range[1],H_range[2]), n_freq_gen) .* exp.(-frequency_decay *  (0:n_freq_gen-1).^2)
    δ = rand(rng, Uniform(0, 2*π), n_freq_gen)

    return ϕ(x,z,0,H,δ)
end



## Generate data
setup_hash = hash((n_sensors,yspan,n_u_trajectories,n_u_trajectories_test,n_u_trajectories_validation,n_y_eval,batch_size,n_freq_gen,frequency_decay,H_range))
data_filename = "small_waves_potential_data_hash_$setup_hash.jld2"

if isfile(data_filename) && !recompute_data
    loaders = FileIO.load(data_filename,"loaders")
    println("Loaded data from disk")
    flush(stdout)
else
    y_locs = generate_y_locs(yspan, n_y_eval, n_u_trajectories+n_u_trajectories_test+n_u_trajectories_validation, equidistant_y)
    loaders = generate_data(x_locs, yspan, u_func, v_func, n_sensors, n_u_trajectories, n_u_trajectories_validation, n_u_trajectories_test, n_y_eval, batch_size; equidistant_y=equidistant_y, y_locs=y_locs)
    if save_on_recompute
        FileIO.save(data_filename,"loaders",loaders)
    end
end

d,s = first(loaders.train)


## Define layers

@assert branch_depth >= 3
branch = Chain(
    Dense(n_sensors, branch_width, activation_function, init=flux_ini),
    [Dense(branch_width, branch_width, activation_function, init=flux_ini) for _ in 1:branch_depth-3]...,
    Dense(branch_width, latent_size, init=flux_ini)
)

@assert trunk_depth >= 3
trunk = Chain(
    Dense(size(yspan,1), trunk_width, activation_function, init=flux_ini),
    [Dense(trunk_width, trunk_width, activation_function, init=flux_ini) for _ in 1:trunk_depth-3]...,
    Dense(trunk_width, latent_size+trunk_var_bias, activation_function, init=flux_ini)
)


##
# Define model
float_type_func = f32
model = DeepONet(trunk=float_type_func(trunk), branch=float_type_func(branch), const_bias_trainable=const_bias_trainable, trunk_var_bias=trunk_var_bias, bias=float_type_func([0.0]))

##



params = Flux.params(model)

loss(((y, u_vals), v_vals), seed) = Flux.mse(model(y,u_vals), v_vals)


println("Loss times:")
@time loss(d,s)
@time loss(d,s)

println("Evaluation times:")
@time model(d[1]...)
@time model(d[1]...)

flush(stdout)


## Training loop
opt = Flux.NAdam()


loss_train = fill(NaN,n_epochs)
loss_validation = fill(NaN,n_epochs)
verbose = 2
file_postfix = ""
model_filename = "models/trained_model_waves_potential$(file_postfix).jld2"
if !load_model
    train!(model, loaders, params, loss, opt, n_epochs, loss_train, loss_validation, verbose)
    if save_model
        FileIO.save(model_filename,"model",model,"loss_train",loss_train,"loss_validation",loss_validation)
    end
else
    loaded_model_file = FileIO.load(model_filename)
    model = loaded_model_file["model"]
    loss_train = loaded_model_file["loss_train"]
    loss_validation = loaded_model_file["loss_validation"]
end


# To be used only after final model is selected
function compute_total_loss(loader)
    loss_test = 0
    for (d,s) in loader
        loss_test+=loss(d,s)/length(loader)
    end
    return loss_test
end
loss_test = compute_total_loss(loaders.test)
# println(@sprintf "Test loss: %.3e" loss_test)

all_v_vec::Vector{Float64} = []
for (d,s) in loaders.test
    append!(all_v_vec, d[2])
end
loss_null_guess=Flux.mse(zeros(size(all_v_vec)...), all_v_vec)
println(@sprintf "Null guess, test loss (pure data): %.3e" loss_null_guess)


flush(stdout)
print("Mean of last $(min(10,n_epochs)) validation errors:\n$(mean(loss_validation[end-min(9,n_epochs-1):end]))")


## Plotting

if do_plots

    plot_seed = n_u_trajectories + n_u_trajectories_validation + n_u_trajectories_test ÷ 2
    x_plot = xi:L/n_sensors:xf
    z_plot = zi:h/50:zf
    y_plot = hcat([[x,z] for x=x_plot for z=z_plot]...)
    u_vals_plot = u_func(x_locs, plot_seed)
    v_vals_plot = reshape(v_func(y_plot, plot_seed), length(z_plot), length(x_plot))
    deepo_solution = reshape(model(y_plot, u_vals_plot)[:], length(z_plot), length(x_plot))
    p1=heatmap(x_plot, z_plot, deepo_solution, reuse = false, title=" ϕ [m²/s] DeepONet\nprediction", clim=extrema([v_vals_plot;deepo_solution]),xticks=[xi,(xi+xf)/2,xf])
    xlabel!("x [m]")
    ylabel!("z [m]")
    title=@sprintf " ϕ [m²/s] error\nMSE %.2e" Flux.mse(deepo_solution, v_vals_plot)
    p2=heatmap(x_plot, z_plot, v_vals_plot-deepo_solution, reuse = false, title=title, yticks=false,xticks=[xi,(xi+xf)/2,xf])
    xlabel!("x [m]")
    p3=heatmap(x_plot, z_plot, v_vals_plot, reuse = false, title=" ϕ [m²/s] analytical\nsolution", clim=extrema([v_vals_plot;deepo_solution]), yticks=false,xticks=[xi,(xi+xf)/2,xf])
    xlabel!("x [m]")
    p = plot(p1, p2, p3, reuse = false, layout = (1,3))
    savefig(p, "plots/small_waves_potential_example.pdf")
    display(p)



    p=plot(loss_train, label="Train", legend=:topright, reuse = false, markershape = :circle, yaxis=:log, title="DeepONet training progress")
    plot!(loss_validation, label="Validation", markershape = :circle)
    xlabel!("Epochs")
    ylabel!("Loss (MSE)")
    savefig(p, "plots/small_waves_potential_training.pdf")
    display(p)

end
