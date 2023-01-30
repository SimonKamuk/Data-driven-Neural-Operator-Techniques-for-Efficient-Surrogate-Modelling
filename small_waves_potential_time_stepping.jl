using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, GraphNeuralNetworks, Statistics, LinearAlgebra, Graphs, SparseArrays, Juno, ForwardDiff, Printf, ProgressBars, Distributions, FFTW, ReverseDiff, FileIO, JLD2, Optimisers

include("MyDeepONet.jl")
using .MyDeepONet


##
# Data setup
L = 50 # base wave length [m]
h = 1/(2*π/L) # water depth, [m]
ρ = 1025 # saltwater density [kg/m^3]
g = 9.82 # acceleration of gravity [m/s^2]
T = L/(sqrt(g*L/(2*π) * tanh(2*π*h/L)))
H_range = [0.1,0.5]
rescale = true
n_u_trajectories = 100
n_u_trajectories_test = 100
n_u_trajectories_validation = 100
n_y_eval = 500
batch_size = 100
n_epochs = 100
n_freq_gen = 1
frequency_decay = 0.1  # Larger number means faster decay, meaning fewer high frequency components
Random.seed!(0)
flux_ini = Flux.glorot_uniform(MersenneTwister(rand(Int64)))
recompute_data = true
save_on_recompute = false
const_bias_trainable = false
trunk_var_bias = true
equidistant_y = false
use_plain_loss = false
file_postfix = "_intermediate"

# Model setup
if length(ARGS) == 0
    n_sensors = 110
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
    physics_weight_initial = 1.0
    regularisation_weight = 1e-4

    save_model = true
    load_model = true

else
    jobindex = parse(Int64, ARGS[1])

    PI_use_AD = false  # AD not currently working
    do_plots = false
end

if rescale
    xi = 0
    xf = 1
    zi = 0
    zf = 1
    ti = 0
    tf = 1
else
    xi = 0
    xf = L
    zi = -h
    zf = 0
    ti = 0
    tf = T
end
yspan = Float64.([xi xf;zi zf; ti tf])

x_locs = hcat(
    [
        rand(Uniform(yspan[i,begin], yspan[i,end]), n_sensors)
        for i in 1:2
    ]...)'




function ϕ(x,z,t,H,δ)

    ϕ_vec = map(1:n_freq_gen) do i
        Li = L/i
        ci = sqrt(g*Li/(2*π) * tanh(2*π*h/Li))
        Ti = Li/ci
        ki = 2*π/Li
        ωi = 2*π/Ti

        return @. -H[i]*ci/2 * cosh(ki*(z+h)) / sinh(ki*h) * sin(ωi*t-ki*x+δ[i])
    end

    return sum(ϕ_vec)
end
function p_excess(x,z,t,H,δ)

    p_vec = map(1:n_freq_gen) do i
        Li = L/i
        ci = sqrt(g*Li/(2*π) * tanh(2*π*h/Li))
        Ti = Li/ci
        ki = 2*π/Li
        ωi = 2*π/Ti

        return @. ρ*g*H[i]/2 * cosh(ki*(z+h)) / sinh(ki*h) * cos(ωi*t-ki*x+δ[i])
    end

    return sum(p_vec)
end
function uw(x,z,t,H,δ)

    uw_vec = map(1:n_freq_gen) do i
        Li = L/i
        ci = sqrt(g*Li/(2*π) * tanh(2*π*h/Li))
        Ti = Li/ci
        ki = 2*π/Li
        ωi = 2*π/Ti

        u = @. π * H[i] / Ti * cosh(ki*(z+h)) / sinh(ki*h) * cos(ωi*t-ki*x+δ[i])
        w = @. - π * H[i] / Ti * sinh(ki*(z+h)) / sinh(ki*h) * sin(ωi*t-ki*x+δ[i])
        return [u,w]
    end

    return hcat(sum(uw_vec)...)'
end

function H_δ(seed)
    rng = MersenneTwister(seed)
    H = rand(rng, Uniform(H_range[1],H_range[2]), n_freq_gen) .* exp.(-frequency_decay *  (0:n_freq_gen-1).^2)
    δ = rand(rng, Uniform(0, 2*π), n_freq_gen)

    return H,δ
end
function u_func(y, seed)
    if rescale
        x = y[1,:]*L
        z = (y[2,:].-1)*h
    else
        x = y[1,:]
        z = y[2,:]
    end

    H,δ = H_δ(seed)

    return ϕ(x,z,ti,H,δ)
end
function v_func(y, seed)
    if rescale
        x = y[1,:]*L
        z = (y[2,:].-1)*h
        t = y[3,:]*T
    else
        x = y[1,:]
        z = y[2,:]
        t = y[3,:]
    end

    H,δ = H_δ(seed)

    return ϕ(x,z,t,H,δ)
end



## Generate data
setup_hash = hash((n_sensors,yspan,n_u_trajectories,n_u_trajectories_test,n_u_trajectories_validation,n_y_eval,batch_size,n_freq_gen,frequency_decay,H_range))
data_filename = "small_waves_potential_timestepping_data_hash_$setup_hash.jld2"

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
    # Dense(size(yspan,1), size(yspan,1), init=flux_ini),
    Dense(size(yspan,1), trunk_width, activation_function, init=flux_ini),
    [Dense(trunk_width, trunk_width, activation_function, init=flux_ini) for _ in 1:trunk_depth-3]...,
    Dense(trunk_width, latent_size+trunk_var_bias, activation_function, init=flux_ini)
)

##
# Define model
float_type_func = f64
model = DeepONet(trunk=float_type_func(trunk), branch=float_type_func(branch), const_bias_trainable=const_bias_trainable, trunk_var_bias=trunk_var_bias, bias=float_type_func([0.0]))

##



params = Flux.params(model)


ϵ = Float64(eps(float_type_func==f32 ? Float32 : Float64)^(1/3))
if rescale
    ϵx_scale = ϵ*L
    ϵz_scale = ϵ*h
    ϵt_scale = ϵ*T
else
    ϵx_scale = ϵ
    ϵz_scale = ϵ
    ϵt_scale = ϵ
end

function eval_trunk_and_combine(model,yy,bb)
    return combine_latent(model,evaluate_trunk(model,yy),bb)
end
function loss_fun_physics_informed(
        ((y, u_vals), v_vals)::Tuple{Tuple{Array{Float64}, Tuple{Array{Float64}}}, Array{Float64}},
        seed::Vector{Int64},
        aux_params::Tuple{DeepONet, Array{Float64}, Float64, Float64, Float64, Float64})

    # return Flux.mse(model(y,u_vals), v_vals)
    model, yspan, ϵ, ϵx_scale, ϵz_scale, ϵt_scale = aux_params

    xi, zi, ti, xf, zf, tf = yspan[:]

    b = evaluate_branch(model,u_vals)

    similar_ones = ones(eltype(y),1,size(y,2))

    preds = eval_trunk_and_combine(model,y,b)
    preds_left = eval_trunk_and_combine(model,[xi * similar_ones ; y[2:3,:]],b)
    preds_right = eval_trunk_and_combine(model,[xf * similar_ones ; y[2:3,:]],b)
    preds_bottom = eval_trunk_and_combine(model,[y[1:1,:]; zi * similar_ones; y[3:3,:]],b)
    preds_top = eval_trunk_and_combine(model,[y[1:1,:] ; zf * similar_ones; y[3:3,:]],b)
    preds_ini = eval_trunk_and_combine(model,[y[1:2,:]; ti * similar_ones],b)
    analytical_ini = hcat(
        map(enumerate(seed)) do (i,s)
            return u_func([y[1:2,i] ; ti], s)
        end...
    )

    preds_left_xpϵ = eval_trunk_and_combine(model, [(xi+ϵ) * similar_ones ; y[2:3,:]],b)
    preds_right_xmϵ = eval_trunk_and_combine(model, [(xf-ϵ) * similar_ones ; y[2:3,:]],b)
    preds_bottom_zpϵ = eval_trunk_and_combine(model, [y[1:1,:]; (zi+ϵ) * similar_ones; y[3:3,:]],b)
    preds_top_zmϵ = eval_trunk_and_combine(model, [y[1:1,:] ; (zf-ϵ) * similar_ones; y[3:3,:]],b)
    preds_top_tpϵ = eval_trunk_and_combine(model, [y[1:1,:] ; zf * similar_ones; y[3:3,:] .+ ϵ],b)
    preds_top_tmϵ = eval_trunk_and_combine(model, [y[1:1,:] ; zf * similar_ones; y[3:3,:] .- ϵ],b)

    preds_p_ϵ00 = eval_trunk_and_combine(model, y .+ [ϵ,0,0],b)
    preds_m_ϵ00 = eval_trunk_and_combine(model, y .- [ϵ,0,0],b)
    preds_p_0ϵ0 = eval_trunk_and_combine(model, y .+ [0,ϵ,0],b)
    preds_m_0ϵ0 = eval_trunk_and_combine(model, y .- [0,ϵ,0],b)

    xx_deriv = (preds_p_ϵ00 + preds_m_ϵ00 - 2 * preds)/ϵx_scale^2
    zz_deriv = (preds_p_0ϵ0 + preds_m_0ϵ0 - 2 * preds)/ϵz_scale^2
    x_deriv_left = (preds_left_xpϵ - preds_left)/ϵx_scale
    x_deriv_right = (preds_right - preds_right_xmϵ)/ϵx_scale
    z_deriv_bottom = (preds_bottom_zpϵ - preds_bottom)/ϵz_scale
    z_deriv_top = (preds_top - preds_top_zmϵ)/ϵz_scale
    tt_deriv_top = (preds_top_tpϵ + preds_top_tmϵ - 2 * preds_top)/ϵt_scale^2

    physics_loss_initial = sum((preds_ini-analytical_ini).^2)
    physics_loss_sides = sum((x_deriv_left-x_deriv_right).^2)
    physics_loss_bottom = sum(z_deriv_bottom.^2)
    physics_loss_top = sum((z_deriv_top + 1/g * tt_deriv_top).^2)
    physics_loss_interior = sum((xx_deriv + zz_deriv).^2)
    data_loss_squared = sum((preds .- v_vals).^2)
    regularisation_loss = sum(norm(Flux.params(model)))

    return (data_loss_squared * data_weight + (physics_loss_top + physics_loss_bottom + physics_loss_sides) * physics_weight_boundary + physics_loss_interior * physics_weight_interior + physics_loss_initial * physics_weight_initial) / (2*batch_size) + regularisation_loss * regularisation_weight
end


aux_params = (model, yspan, ϵ, ϵx_scale, ϵz_scale, ϵt_scale)

loss_fun_plain(((y, u_vals), v_vals), seed, aux_params) = Flux.mse(model(y,u_vals), v_vals)

println("Loss times:")
@time loss_fun_physics_informed(d,s,aux_params)
@time loss_fun_physics_informed(d,s,aux_params)

println("Evaluation times:")
@time model(d[1]...)
@time model(d[1]...)

flush(stdout)


## Training loop
opt = Flux.NAdam()

if use_plain_loss
    loss = loss_fun_plain
else
    loss = loss_fun_physics_informed
end


loss_train = fill(NaN,n_epochs)
loss_validation = fill(NaN,n_epochs)
verbose = 2

model_filename = "models/trained_model_waves_timestepping$(file_postfix).jld2"
if !load_model | !isfile(model_filename)
    train!(model, loaders, params, loss, opt, n_epochs, loss_train, loss_validation, verbose, loss_fun_plain, aux_params)
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
        loss_test+=loss_fun_plain(d,s,aux_params)/length(loader)
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
    y_t0_plot = hcat([[x,z,ti] for x=x_plot for z=z_plot]...)
    y_t1_plot = hcat([[x,z,tf/2] for x=x_plot for z=z_plot]...)

    u_vals_plot = u_func(x_locs, plot_seed)
    input_fun_plot = reshape(u_func(y_t0_plot, plot_seed), length(z_plot), length(x_plot))
    v_vals_plot = reshape(v_func(y_t1_plot, plot_seed), length(z_plot), length(x_plot))
    deepo_solution = reshape(model(y_t1_plot, u_vals_plot)[:], length(z_plot), length(x_plot))

    xticks = [xi,(xi+xf)/2,xf]
    if rescale
        x_plot = x_plot*L
        z_plot = (z_plot.-1)*h
        xticks = xticks*L
        scaled_x_locs = (x_locs .- [0;1]) .* [L;h]
    else
        scaled_x_locs = x_locs
    end

    p1=heatmap(x_plot, z_plot, deepo_solution, reuse = false, title="DeepONet\nprediction ϕ [m²/s]", clim=extrema([v_vals_plot;deepo_solution]),xticks=xticks)
    xlabel!("x [m]")
    ylabel!("z [m]")
    title=@sprintf "Error ϕ [m²/s]\nMSE %.2e" Flux.mse(deepo_solution, v_vals_plot)
    p2=heatmap(x_plot, z_plot, v_vals_plot-deepo_solution, reuse = false, title=title, yticks=false,xticks=xticks)
    xlabel!("x [m]")
    p3=heatmap(x_plot, z_plot, v_vals_plot, reuse = false, title="Analytical\nsolution ϕ [m²/s]", clim=extrema([v_vals_plot;deepo_solution]), yticks=false,xticks=xticks)
    xlabel!("x [m]")
    p = plot(p1, p2, p3, reuse = false, layout = (1,3))
    savefig(p, "plots/small_waves_potential_timestepping_example$(file_postfix).pdf")
    display(p)


    title=@sprintf "ϕ [m²/s] error at t=T/2, MSE %.2e" Flux.mse(deepo_solution, v_vals_plot)
    p=plot(x_plot, z_plot, v_vals_plot-deepo_solution, reuse = false, title=title ,xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_potential_timestepping_example_3d_error$(file_postfix).pdf")
    display(p)

    p=plot(x_plot, z_plot, deepo_solution, reuse = false, title="ϕ [m²/s] DeepONet prediction at t=T/2", clim=extrema([v_vals_plot;deepo_solution]),xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_potential_timestepping_example_3d_pred$(file_postfix).pdf")
    display(p)

    p=plot(x_plot, z_plot, v_vals_plot, reuse = false, title="ϕ [m²/s] analytical solution at t=T/2", clim=extrema([v_vals_plot;deepo_solution]), xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_potential_timestepping_example_3d_analytical$(file_postfix).pdf")
    display(p)

    p=plot(x_plot, z_plot, input_fun_plot, reuse = false, title="ϕ [m²/s] input function (t=0 T)", clim=extrema([v_vals_plot;deepo_solution]), xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    scatter!(scaled_x_locs[1,:],scaled_x_locs[2,:],u_vals_plot,color=:black,label="Sensors")
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_potential_timestepping_example_3d_input_fun$(file_postfix).pdf")
    display(p)



    p=plot(loss_train, label="Train", legend=:topright, reuse = false, markershape = :circle, yaxis=:log, title="DeepONet training progress")
    plot!(loss_validation, label="Validation", markershape = :circle)
    xlabel!("Epochs")
    ylabel!("Loss (MSE)")
    savefig(p, "plots/small_waves_potential_timestepping_training$(file_postfix).pdf")
    display(p)


    ## Loss vs time step
    times = zeros(n_u_trajectories_test*n_y_eval)
    Hs = zeros(n_u_trajectories_test*n_y_eval)
    δs = zeros(n_u_trajectories_test*n_y_eval)
    losses = zeros(n_u_trajectories_test*n_y_eval)
    for (batch_id,(d,s)) in enumerate(loaders.test)
        ((y, u_vals), v_vals) = d
        for i in 1:batch_size
            H,δ = H_δ(s[i])
            Hs[(batch_id-1)*batch_size + i] = H[]
            δs[(batch_id-1)*batch_size + i] = δ[]
            times[(batch_id-1)*batch_size + i] = y[3,i]
            losses[(batch_id-1)*batch_size + i] = loss_fun_plain(((y[:,i:i], (u_vals[1][:,i:i],)), v_vals[1:1,i:i]),s[i],aux_params)
        end
    end

    pyplot_hexbin_times_inputs = (times,losses,(0:0.25:1, ["0 T", "0.25 T", "0.5 T", "0.75 T", "T"]),"Loss vs. time for test set","Time (unit of wave periods)","Squared error","plots/small_waves_potential_timestepping_loss_vs_time$(file_postfix).pdf")
    pyplot_hexbin_H_inputs = (Hs,losses,([0.1,0.2,0.3,0.4,0.5],),"Loss vs. amplitude for test set","Amplitude, H [m]","Squared error","plots/small_waves_potential_timestepping_loss_vs_H$(file_postfix).pdf")
    pyplot_hexbin_delta_inputs = (δs,losses,([0,π,2π],["0","π","2π"]),"Loss vs. phase for test set","Phase, δ","Squared error","plots/small_waves_potential_timestepping_loss_vs_delta$(file_postfix).pdf")

    FileIO.save("hexbin_plot_data_fixed_depth.jld2","pyplot_hexbin_times_inputs",pyplot_hexbin_times_inputs,"pyplot_hexbin_H_inputs",pyplot_hexbin_H_inputs,"pyplot_hexbin_delta_inputs",pyplot_hexbin_delta_inputs)

end
