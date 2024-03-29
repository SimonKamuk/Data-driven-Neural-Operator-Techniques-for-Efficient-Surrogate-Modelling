using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, GraphNeuralNetworks, Statistics, LinearAlgebra, Graphs, SparseArrays, Juno, ForwardDiff, Printf, ProgressBars, Distributions, FFTW, ReverseDiff, FileIO, JLD2, Optimisers

include("MyDeepONet.jl")
using .MyDeepONet



# Data setup
L = 50 # base wave length [m]
h = 1/(2*π/L) # water depth, [m]
ρ = 1025 # saltwater density [kg/m^3]
g = 9.82 # acceleration of gravity [m/s^2]
H_range = [0.1,0.5]
rescale = true
if rescale
    xi = 0
    xf = 1
    zi = 0
    zf = 1
else
    xi = 0
    xf = L
    zi = -h
    zf = 0
end
yspan = [xi xf;zi zf]
n_u_trajectories = 25
n_u_trajectories_test = 100
n_u_trajectories_validation = 100
n_y_eval = 200
batch_size = 100
n_epochs = 1000
n_freq_gen = 1
frequency_decay = 0.1  # Larger number means faster decay, meaning fewer high frequency components
Random.seed!(0)
flux_ini = Flux.glorot_uniform(MersenneTwister(rand(Int64)))
recompute_data = true
save_on_recompute = false
const_bias_trainable = false
trunk_var_bias = true
equidistant_y = false
file_postfix = "_intermediate"

# Model setup
if length(ARGS) == 0
    n_sensors = 50
    branch_width = 40
    trunk_width = 70
    branch_depth = 3
    trunk_depth = 5
    latent_size = 75
    activation_function = softplus
    do_plots = true

    data_weight = 1.0
    physics_weight_boundary = 1.0
    physics_weight_interior = 1.0
    regularisation_weight = 1e-4

    save_model = true
    load_model = true

else
    jobindex = parse(Int64, ARGS[1])

    PI_use_AD = false  # AD not currently working
    do_plots = false
end



x_locs = range(start=xi, stop=xf, length=n_sensors+1)[begin:end-1]
x_locs_full = cat(x_locs,xf,dims=1)


function ϕ(x,z,t,H,δ)

    ϕ_vec = map(1:n_freq_gen) do i
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
        c = sqrt(g*L/(2*π*i) * tanh(2*π*h*i/L))
        T = L/(c*i)
        k = 2*π*i/L
        ω = 2*π/T

        u = @. π * H[i] / T * cosh(k*(z+h)) / sinh(k*h) * cos(ω*t-k*x+δ[i])
        w = @. - π * H[i] / T * sinh(k*(z+h)) / sinh(k*h) * sin(ω*t-k*x+δ[i])
        return [u,w]
    end

    return hcat(sum(uw_vec)...)'
end


function u_func(x, seed)
    # Define input function
    rng = MersenneTwister(seed)
    H = rand(rng, Uniform(H_range[1],H_range[2]), n_freq_gen) .* exp.(-frequency_decay *  (0:n_freq_gen-1).^2)
    δ = rand(rng, Uniform(0, 2*π), n_freq_gen)

    if rescale
        return ϕ(x*L,0,0,H,δ)
    else
        return ϕ(x,0,0,H,δ)
    end
end
function v_func(y, seed)
    if rescale
        x = y[1,:]*L
        z = (y[2,:].-1)*h
    else
        x = y[1,:]
        z = y[2,:]
    end

    rng = MersenneTwister(seed)
    H = rand(rng, Uniform(H_range[1],H_range[2]), n_freq_gen) .* exp.(-frequency_decay *  (0:n_freq_gen-1).^2)
    δ = rand(rng, Uniform(0, 2*π), n_freq_gen)

    return uw(x,z,0,H,δ)
end



## Generate data
setup_hash = hash((n_sensors,yspan,n_u_trajectories,n_u_trajectories_test,n_u_trajectories_validation,n_y_eval,batch_size,n_freq_gen,frequency_decay,H_range))
data_filename = "small_waves_velocity_data_hash_$setup_hash.jld2"

if isfile(data_filename) && !recompute_data
    loaders = FileIO.load(data_filename,"loaders")
    println("Loaded data from disk")
    flush(stdout)
else
    y_locs = generate_y_locs(yspan, n_y_eval, n_u_trajectories+n_u_trajectories_test+n_u_trajectories_validation, equidistant_y)
    loaders = generate_data(x_locs, yspan, u_func, v_func, n_sensors, n_u_trajectories, n_u_trajectories_validation, n_u_trajectories_test, n_y_eval, batch_size; equidistant_y=equidistant_y, y_locs=y_locs, vdim=2)
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
trunk = [
    Chain(
        Dense(size(yspan,1), trunk_width, activation_function, init=flux_ini),
        [Dense(trunk_width, trunk_width, activation_function, init=flux_ini) for _ in 1:trunk_depth-3]...,
        Dense(trunk_width, latent_size+trunk_var_bias, activation_function, init=flux_ini)
    ),
    Chain(
        Dense(size(yspan,1), trunk_width, activation_function, init=flux_ini),
        [Dense(trunk_width, trunk_width, activation_function, init=flux_ini) for _ in 1:trunk_depth-3]...,
        Dense(trunk_width, latent_size+trunk_var_bias, activation_function, init=flux_ini)
    )
]


##
# Define model
float_type_func = f32
model = DeepONet(trunk=float_type_func(trunk), branch=float_type_func(branch), const_bias_trainable=const_bias_trainable, trunk_var_bias=trunk_var_bias, bias=float_type_func([0.0]))

##



params = Flux.params(model)

function eval_trunk_and_combine(yy,bb)
    return combine_latent(model,evaluate_trunk(model,yy),bb)
end
ϵ = Float64(eps(float_type_func==f32 ? Float32 : Float64)^(1/3))
function loss(((y, u_vals), v_vals), seed)
    # y = [y[1:1,:]*L;(y[2:2,:].-1)*h]

    b = evaluate_branch(model,u_vals)

    similar_ones = ones(eltype(y),1,size(y,2))

    t = evaluate_trunk(model,y)
    t_left = evaluate_trunk(model,[xi * similar_ones ; y[2:2,:]])
    t_right = evaluate_trunk(model,[xf * similar_ones ; y[2:2,:]])
    t_bottom = evaluate_trunk(model,[y[1:1,:]; zi * similar_ones])
    t_top = evaluate_trunk(model,[y[1:1,:] ; zf * similar_ones])

    uw_left = combine_latent(model,t_left,b)
    uw_right = combine_latent(model,t_right,b)
    uw_bottom = combine_latent(model,t_bottom,b)
    uw_top = combine_latent(model,t_top,b)

    analytical_top = hcat(
        map(enumerate(seed)) do (i,s)
            return v_func([y[1,i] ; zf], s)
        end...
    )

    preds = combine_latent(model,t,b)

    preds_p_ϵ0 = eval_trunk_and_combine(y .+ [ϵ,0],b)
    preds_m_ϵ0 = eval_trunk_and_combine(y .- [ϵ,0],b)
    preds_p_0ϵ = eval_trunk_and_combine(y .+ [0,ϵ],b)
    preds_m_0ϵ = eval_trunk_and_combine(y .- [0,ϵ],b)

    if rescale
        v1_y1_deriv = (preds_p_ϵ0[1,:] .- preds_m_ϵ0[1,:])/(L*2*ϵ)
        v2_y2_deriv = (preds_p_0ϵ[2,:] .- preds_m_0ϵ[2,:])/(h*2*ϵ)
    else
        v1_y1_deriv = (preds_p_ϵ0[1,:] .- preds_m_ϵ0[1,:])/(2*ϵ)
        v2_y2_deriv = (preds_p_0ϵ[2,:] .- preds_m_0ϵ[2,:])/(2*ϵ)
    end
    physics_loss_interior = sum((v1_y1_deriv + v2_y2_deriv).^2)

    physics_loss_sides = sum((uw_left[1,:]-uw_right[1,:]).^2)
    physics_loss_bottom = sum(uw_bottom[2,:].^2)
    physics_loss_top = sum((uw_top .- analytical_top).^2)
    data_loss_squared = sum((preds .- v_vals).^2)
    regularisation_loss = sum(norm(Flux.params(model)))

    return (data_loss_squared * data_weight + (physics_loss_top + physics_loss_bottom + physics_loss_sides) * physics_weight_boundary + physics_loss_interior * physics_weight_interior) / (2*batch_size) + regularisation_loss * regularisation_weight
end


loss_fun_plain(((y, u_vals), v_vals), seed) = Flux.mse(model(y,u_vals), v_vals)

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
model_filename = "models/trained_model_waves_velocity$(file_postfix).jld2"
if !load_model | !isfile(model_filename)
    train!(model, loaders, params, loss, opt, n_epochs, loss_train, loss_validation, verbose, loss_fun_plain)
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
        loss_test+=loss_fun_plain(d,s)/length(loader)
    end
    return loss_test
end
loss_test_no_phys = compute_total_loss(loaders.test)
loss_val_no_phys = compute_total_loss(loaders.validation)
println(@sprintf "Test loss (pure data): %.3e" loss_test_no_phys)
println(@sprintf "Validation loss (pure data): %.3e" loss_val_no_phys)


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
    x_plot = xi:(xf-xi)/50:xf
    z_plot = zi:(zf-zi)/50:zf
    y_plot = hcat([[x,z] for x=x_plot for z=z_plot]...)
    u_vals_plot = u_func(x_locs, plot_seed)
    v_vals_plot = reshape(v_func(y_plot, plot_seed), 2, length(z_plot), length(x_plot))
    deepo_solution = reshape(model(y_plot, u_vals_plot)[:], 2, length(z_plot), length(x_plot))
    xticks = [xi,(xi+xf)/2,xf]

    if rescale
        x_plot = x_plot*L
        z_plot = (z_plot.-1)*h
        xticks = xticks*L
    end

    p1=heatmap(x_plot, z_plot, deepo_solution[1,:,:], reuse = false, title="u, DeepONet\nprediction", clim=extrema([v_vals_plot;deepo_solution]),xticks=xticks)
    xlabel!("x [m]")
    ylabel!("z [m]")
    title=@sprintf "u [m/s], Error\nMSE %.2e" Flux.mse(deepo_solution[1,:,:], v_vals_plot[1,:,:])
    p2=heatmap(x_plot, z_plot, v_vals_plot[1,:,:]-deepo_solution[1,:,:], reuse = false, title=title, yticks=false,xticks=xticks)
    xlabel!("x [m]")
    p3=heatmap(x_plot, z_plot, v_vals_plot[1,:,:], reuse = false, title="u, Analytical\nsolution", clim=extrema([v_vals_plot;deepo_solution]), yticks=false,xticks=xticks)
    xlabel!("x [m]")
    p = plot(p1, p2, p3, reuse = false, layout = (1,3))
    savefig(p, "plots/small_waves_u_velocity_example$(file_postfix).pdf")
    display(p)

    p1=heatmap(x_plot, z_plot, deepo_solution[2,:,:], reuse = false, title="w, DeepONet\nprediction", clim=extrema([v_vals_plot;deepo_solution]),xticks=xticks)
    xlabel!("x [m]")
    ylabel!("z [m]")
    title=@sprintf "w [m/s], Error\nMSE %.2e" Flux.mse(deepo_solution[2,:,:], v_vals_plot[2,:,:])
    p2=heatmap(x_plot, z_plot, v_vals_plot[2,:,:]-deepo_solution[2,:,:], reuse = false, title=title, yticks=false,xticks=xticks)
    xlabel!("x [m]")
    p3=heatmap(x_plot, z_plot, v_vals_plot[2,:,:], reuse = false, title="w, Analytical\nsolution", clim=extrema([v_vals_plot;deepo_solution]), yticks=false,xticks=xticks)
    xlabel!("x [m]")
    p = plot(p1, p2, p3, reuse = false, layout = (1,3))
    savefig(p, "plots/small_waves_w_velocity_example$(file_postfix).pdf")
    display(p)



    p=plot(x_plot, z_plot, deepo_solution[1,:,:], reuse = false, title="DeepONet prediction, u [m/s]", clim=extrema([v_vals_plot;deepo_solution]),xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_u_velocity_example_3d_pred$(file_postfix).pdf")
    display(p)

    title=@sprintf "Error, u [m/s], MSE %.2e" Flux.mse(deepo_solution[1,:,:], v_vals_plot[1,:,:])
    p=plot(x_plot, z_plot, v_vals_plot[1,:,:]-deepo_solution[1,:,:], reuse = false, title=title ,xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_u_velocity_example_3d_error$(file_postfix).pdf")
    display(p)

    p=plot(x_plot, z_plot, v_vals_plot[1,:,:], reuse = false, title="Analytical solution, u [m/s]", clim=extrema([v_vals_plot;deepo_solution]), xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_u_velocity_example_3d_analytical$(file_postfix).pdf")
    display(p)


    p=plot(x_plot, z_plot, deepo_solution[2,:,:], reuse = false, title="DeepONet prediction, w [m/s]", clim=extrema([v_vals_plot;deepo_solution]),xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_w_velocity_example_3d_pred$(file_postfix).pdf")
    display(p)

    title=@sprintf "Error, w [m/s], MSE %.2e" Flux.mse(deepo_solution[2,:,:], v_vals_plot[2,:,:])
    p=plot(x_plot, z_plot, v_vals_plot[2,:,:]-deepo_solution[2,:,:], reuse = false, title=title, xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_w_velocity_example_3d_error$(file_postfix).pdf")
    display(p)

    p=plot(x_plot, z_plot, v_vals_plot[2,:,:], reuse = false, title="Analytical, solution, w [m/s]", clim=extrema([v_vals_plot;deepo_solution]), xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_w_velocity_example_3d_analytical$(file_postfix).pdf")
    display(p)



    p=plot(loss_train, label="Train", legend=:topright, reuse = false, markershape = :circle, yaxis=:log, title="DeepONet training progress", ylims=[1e-6,1])
    plot!(loss_validation, label="Validation", markershape = :circle)
    xlabel!("Epochs")
    ylabel!("Loss (MSE)")
    savefig(p, "plots/small_waves_velocity_training$(file_postfix).pdf")
    display(p)

end
