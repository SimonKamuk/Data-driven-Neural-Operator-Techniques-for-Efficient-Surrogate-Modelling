using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, Statistics, LinearAlgebra, Juno, ForwardDiff, Printf, ProgressBars, Distributions, FFTW, ReverseDiff, FileIO, JLD2, Optimisers, Profile

include("MyDeepONet.jl")
using .MyDeepONet


##
# Data setup
const L::Float64 = 50.0 # base wave length [m]
const ρ::Int64 = 1025 # saltwater density [kg/m^3]
const g::Float64 = 9.82 # acceleration of gravity [m/s^2]
const kh_range::Vector{Float64} = [0.4,4]
const H_range::Vector{Float64} = [0.1,0.5]
const n_u_trajectories::Int64 = 300
const n_u_trajectories_test::Int64 = 300
const n_u_trajectories_validation::Int64 = 300
const n_y_eval::Int64 = 250
const batch_size::Int64 = 100
const frequency_decay::Float64 = 0.1  # Larger number means faster decay, meaning fewer high frequency components
Random.seed!(0)
flux_ini = Flux.glorot_uniform(MersenneTwister(abs(rand(Int64))))
const recompute_data::Bool = true
const save_on_recompute::Bool = false
const const_bias_trainable::Bool = false
const trunk_var_bias::Bool = true
const equidistant_y::Bool = false
const use_plain_loss::Bool = false



# Model setup
if length(ARGS) == 0
    const n_epochs::Int64 = 100
    const n_sensors::Int64 = 110
    const branch_width::Int64 = 40
    const trunk_width::Int64 = 70
    const branch_depth::Int64 = 3
    const trunk_depth::Int64 = 5
    const latent_size::Int64 = 75
    const activation_function = softplus
    const do_plots::Bool = true

    const data_weight::Float64 = 1.0
    const physics_weight_boundary::Float64 = 1.0
    const physics_weight_interior::Float64 = 1.0
    const physics_weight_initial::Float64 = 1.0
    const regularisation_weight::Float64 = 1e-4

    const n_freq_gen::Int64 = 1
    file_postfix::String = n_freq_gen == 1 ? "" : "_polychrome"

    const save_model::Bool = true
    const load_model::Bool = false

else
    jobindex = parse(Int64, ARGS[1])
    const do_plots::Bool = false

    const n_freq_gen::Int64 = 100
    const n_epochs::Int64 = 200

    (par_n_sensors,par_trunk_width,par_branch_width,par_latent_size,par_branch_depth,par_trunk_depth) = [
    (n_sensors,trunk_width,branch_width,latent_size,branch_depth,trunk_depth)
    for n_sensors in [50,100,150]
    for trunk_width in [50,75,100]
    for branch_width in [50,75,100]
    for latent_size in [50,75,100]
    for branch_depth in [3,4,5]
    for trunk_depth in [4,5,6,7]
    ][jobindex]

    const n_sensors::Int64 = par_n_sensors
    const branch_width::Int64 = par_branch_width
    const trunk_width::Int64 = par_trunk_width
    const branch_depth::Int64 = par_branch_depth
    const trunk_depth::Int64 = par_trunk_depth
    const latent_size::Int64 = par_latent_size

    const activation_function = softplus


    const data_weight::Float64 = 1.0
    const physics_weight_boundary::Float64 = 1.0
    const physics_weight_interior::Float64 = 1.0
    const physics_weight_initial::Float64 = 1.0
    const regularisation_weight::Float64 = 1e-4

    file_postfix::String = ""

    const save_model::Bool = false
    const load_model::Bool = false

end


xi = 0
xf = 1
zi = 0
zf = 1
ti = 0
tf = 1

const yspan::Matrix{Float64} = Float64.([xi xf;zi zf; ti tf])

# const x_locs::Matrix{Float64} = hcat(
#     [
#         rand(Uniform(yspan[i,begin], yspan[i,end]), n_sensors)
#         for i in 1:2
#     ]...)'
const x_locs::Matrix{Float64} = hcat(
    rand(Uniform(yspan[1,begin], yspan[1,end]), n_sensors),
    fill(zf,n_sensors)
)'


const L_vec::Matrix{Float64} = reshape([L/i for i in 1:n_freq_gen],1,:)
const k_vec::Matrix{Float64} = reshape([2*π/l for l in L_vec],1,:)
const c_prefactor::Matrix{Float64} = reshape([sqrt( g*l/(2*π) ) for l in L_vec],1,:)

function ϕ(x,z,t,H,δ,h)
    c_vec = c_prefactor .* sqrt.( tanh.(2*π*h./L_vec) )
    T_vec = L_vec./c_vec
    ω_vec = 2*π./T_vec

    ϕ_vec = -H.*c_vec/2 .* cosh.(k_vec.*(z.+h)) ./ sinh.(k_vec.*h) .* sin.(ω_vec.*t.-k_vec.*x.+δ)

    return sum(ϕ_vec, dims=2)[:,1]
end
function p_excess(x,z,t,H,δ,h)

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
function uw(x,z,t,H,δ,h)

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

const rng_vec::Vector{MersenneTwister} = MersenneTwister.(1:n_u_trajectories + n_u_trajectories_validation + n_u_trajectories_test)
const freq_decay_coef_vec::Vector{Float64} = exp.(-frequency_decay *  (0:n_freq_gen-1).^2)
function H_δ_kh(seed)
    n = length(seed)
    vecs = mapreduce(hcat,1:n) do i
        rng = copy(rng_vec[seed[i]])
        # rng = MersenneTwister(seed[i])
        H = rand(rng, Uniform(H_range[1],H_range[2]), n_freq_gen) .* freq_decay_coef_vec
        δ = rand(rng, Uniform(0, 2*π), n_freq_gen)
        kh = rand(rng, Uniform(kh_range[1],kh_range[2]),1)
        return [H,δ,kh]
    end


    H = reduce(hcat,vecs[1,:])'
    δ = reduce(hcat,vecs[2,:])'
    kh = reduce(hcat,vecs[3,:])'

    return H,δ,kh
end
function u_func(y, seed)
    H,δ,kh = H_δ_kh(seed)

    h = kh/(2*π/L) # water depth, [m]

    x = y[1,:]*L
    z = (y[2,:].-1).*h

    return ϕ(x,z,ti,H,δ,h)
end
function u_func(y, H,δ,kh)
    h = kh/(2*π/L) # water depth, [m]

    x = y[1,:]*L
    z = (y[2,:].-1).*h

    return ϕ(x,z,ti,H,δ,h)
end
function v_func(y, seed)
    H,δ,kh = H_δ_kh(seed)

    h = kh/(2*π/L) # water depth, [m]
    T = L./(sqrt.(g*L/(2*π) * tanh.(2*π*h/L)))

    x = y[1,:]*L
    z = (y[2,:].-1).*h
    t = y[3,:].*T

    return ϕ(x,z,t,H,δ,h)
end



# Old function definitions
# function ϕ(x,z,t,H,δ)
#
#     ϕ_vec = map(1:n_freq_gen) do i
#         Li = L/i
#         ci = sqrt(g*Li/(2*π) * tanh(2*π*h/Li))
#         Ti = Li/ci
#         ki = 2*π/Li
#         ωi = 2*π/Ti
#
#         return @. -H[i]*ci/2 * cosh(ki*(z+h)) / sinh(ki*h) * sin(ωi*t-ki*x+δ[i])
#     end
#
#     return sum(ϕ_vec)
# end
# function H_δ_kh(seed)
#     rng = MersenneTwister(seed)
#     H = rand(rng, Uniform(H_range[1],H_range[2]), n_freq_gen) .* exp.(-frequency_decay *  (0:n_freq_gen-1).^2)
#     δ = rand(rng, Uniform(0, 2*π), n_freq_gen)
#     kh = rand(rng, Uniform(kh_range[1],kh_range[2]),1)
#
#     return H,δ,kh
# end

## Generate data
setup_hash = hash((n_sensors,yspan,n_u_trajectories,n_u_trajectories_test,n_u_trajectories_validation,n_y_eval,batch_size,n_freq_gen,frequency_decay,H_range))
data_filename = "small_waves_surface_potential_timestepping_var_depth_data_hash_$setup_hash.jld2"

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
    Dense(size(yspan,1)+1, trunk_width, activation_function, init=flux_ini),
    [Dense(trunk_width, trunk_width, activation_function, init=flux_ini) for _ in 1:trunk_depth-3]...,
    Dense(trunk_width, latent_size+trunk_var_bias, activation_function, init=flux_ini)
)

##
# Define model
float_type_func = f64
model = DeepONet(trunk=float_type_func(trunk), branch=float_type_func(branch), const_bias_trainable=const_bias_trainable, trunk_var_bias=trunk_var_bias, bias=float_type_func([0.0]))

##



params = Flux.params(model)
const ϵ::Float64 = Float64(eps(float_type_func==f32 ? Float32 : Float64)^(1/3))



function eval_trunk_and_combine(model::DeepONet,yy::Matrix{Float64},bb::Matrix{Float64})
    return combine_latent(model,evaluate_trunk(model,yy),bb)
end
function loss_fun_physics_informed(
        ((y, u_vals), v_vals)::Tuple{Tuple{Matrix{Float64}, Tuple{Matrix{Float64}}}, Matrix{Float64}},
        seed::Vector{Int64},
        model::DeepONet)

    H_vec,δ_vec,kh_vec = H_δ_kh(seed)
    h_vec = kh_vec[:,1] ./ (2*π/L)#[H_δ_kh(s)[3][1]/(2*π/L) for s in seed]
    T_vec = ((g/(L*2*π)) .* tanh.((2*π/L).*h_vec)).^(-1/2)

    ϵx_scale = ϵ*L
    ϵz_scale = ϵ*h_vec'
    ϵt_scale = ϵ*T_vec'

    xi, zi, ti, xf, zf, tf = yspan[:]

    b = evaluate_branch(model,u_vals)

    # similar_ones = ones(eltype(y),1,size(y,2))
    analytical_ini = u_func([y[1:2,:] ; fill(ti, 1, batch_size)], H_vec,δ_vec,kh_vec)'

    # preds = eval_trunk_and_combine(model,y,b)
    # preds_left = eval_trunk_and_combine(model,[xi * similar_ones ; y[2:3,:]],b)
    # preds_right = eval_trunk_and_combine(model,[xf * similar_ones ; y[2:3,:]],b)
    # preds_bottom = eval_trunk_and_combine(model,[y[1:1,:]; zi * similar_ones; y[3:3,:]],b)
    # preds_top = eval_trunk_and_combine(model,[y[1:1,:] ; zf * similar_ones; y[3:3,:]],b)
    # preds_ini = eval_trunk_and_combine(model,[y[1:2,:]; ti * similar_ones],b)
    # preds_left_xpϵ = eval_trunk_and_combine(model, [(xi+ϵ) * similar_ones ; y[2:3,:]],b)
    # preds_right_xmϵ = eval_trunk_and_combine(model, [(xf-ϵ) * similar_ones ; y[2:3,:]],b)
    # preds_bottom_zpϵ = eval_trunk_and_combine(model, [y[1:1,:]; (zi+ϵ) * similar_ones; y[3:3,:]],b)
    # preds_top_zmϵ = eval_trunk_and_combine(model, [y[1:1,:] ; (zf-ϵ) * similar_ones; y[3:3,:]],b)
    # preds_top_tpϵ = eval_trunk_and_combine(model, [y[1:1,:] ; zf * similar_ones; y[3:3,:] .+ ϵ],b)
    # preds_top_tmϵ = eval_trunk_and_combine(model, [y[1:1,:] ; zf * similar_ones; y[3:3,:] .- ϵ],b)
    # preds_p_ϵ00 = eval_trunk_and_combine(model, y .+ [ϵ,0,0],b)
    # preds_m_ϵ00 = eval_trunk_and_combine(model, y .- [ϵ,0,0],b)
    # preds_p_0ϵ0 = eval_trunk_and_combine(model, y .+ [0,ϵ,0],b)
    # preds_m_0ϵ0 = eval_trunk_and_combine(model, y .- [0,ϵ,0],b)

    y = [y;kh_vec']
    all_y = hcat(
    y,
    [fill(xi, 1, batch_size) ; y[2:4,:]],
    [fill(xf, 1, batch_size) ; y[2:4,:]],
    [y[1:1,:]; fill(zi, 1, batch_size); y[3:4,:]],
    [y[1:1,:] ; fill(zf, 1, batch_size); y[3:4,:]],
    [y[1:2,:]; fill(ti, 1, batch_size); y[4:4,:]],
    [fill(xi+ϵ, 1, batch_size) ; y[2:4,:]],
    [fill(xf-ϵ, 1, batch_size) ; y[2:4,:]],
    [y[1:1,:]; fill(zi+ϵ, 1, batch_size); y[3:4,:]],
    [y[1:1,:] ; fill(zf-ϵ, 1, batch_size); y[3:4,:]],
    [y[1:1,:] ; fill(zf, 1, batch_size); y[3:4,:] .+ ϵ],
    [y[1:1,:] ; fill(zf, 1, batch_size); y[3:4,:] .- ϵ],
    y .+ [ϵ,0,0,0],
    y .- [ϵ,0,0,0],
    y .+ [0,ϵ,0,0],
    y .- [0,ϵ,0,0],
    )

    all_b = hcat([b for _ in 1:16]...)
    all_preds = eval_trunk_and_combine(model, all_y, all_b)

    (
    preds,
    preds_left,
    preds_right,
    preds_bottom,
    preds_top,
    preds_ini,
    preds_left_xpϵ,
    preds_right_xmϵ,
    preds_bottom_zpϵ,
    preds_top_zmϵ,
    preds_top_tpϵ,
    preds_top_tmϵ,
    preds_p_ϵ00,
    preds_m_ϵ00,
    preds_p_0ϵ0,
    preds_m_0ϵ0,
    ) = [view(all_preds, :, batch_size*(i-1)+1:batch_size*i) for i = 1:16]



    xx_deriv = (preds_p_ϵ00 + preds_m_ϵ00 - 2 * preds)./ϵx_scale.^2
    zz_deriv = (preds_p_0ϵ0 + preds_m_0ϵ0 - 2 * preds)./ϵz_scale.^2
    x_deriv_left = (preds_left_xpϵ - preds_left)./ϵx_scale
    x_deriv_right = (preds_right - preds_right_xmϵ)./ϵx_scale
    z_deriv_bottom = (preds_bottom_zpϵ - preds_bottom)./ϵz_scale
    z_deriv_top = (preds_top - preds_top_zmϵ)./ϵz_scale
    tt_deriv_top = (preds_top_tpϵ + preds_top_tmϵ - 2 * preds_top)./ϵt_scale.^2

    physics_loss_initial = sum(abs2, preds_ini-analytical_ini)
    physics_loss_sides = sum(abs2, x_deriv_left-x_deriv_right)
    physics_loss_bottom = sum(abs2, z_deriv_bottom)
    physics_loss_top = sum(abs2, z_deriv_top + 1/g * tt_deriv_top)
    physics_loss_interior = sum(abs2, xx_deriv + zz_deriv)
    data_loss_squared = sum(abs2, preds .- v_vals)
    regularisation_loss = sum(norm(Flux.params(model)))

    return (data_loss_squared * data_weight + (physics_loss_top + physics_loss_bottom + physics_loss_sides) * physics_weight_boundary + physics_loss_interior * physics_weight_interior + physics_loss_initial * physics_weight_initial) / (2*batch_size) + regularisation_loss * regularisation_weight
end

function period_prediction(d,s)
    ((y, u_vals), v_vals) = d
    u_vals_this_time = repeat(u_vals[1],inner=(1,n_sensors))
    kh = fill(0.4,batch_size,1)
    u_vals_this_time = model([repeat(x_locs,1,batch_size);fill(1,1,batch_size*n_sensors);repeat(kh',inner=(1,n_sensors))],(u_vals_this_time,))
    u_vals_this_time = repeat(reshape(u_vals_this_time,n_sensors,batch_size),inner=(1,n_sensors))
    return u_vals_this_time
end

##
println("Loss times per batch (100 repetitions):")
@time [loss_fun_physics_informed(d,s,model) for _ in 1:100]
@time [loss_fun_physics_informed(d,s,model) for _ in 1:100]

println("Evaluation times per batch:")
args = ([d[1][1];fill(0.4,1,batch_size)], d[1][2])
@time model(args...)
@time model(args...)



println("Evaluation times per period per batch:")
@time period_prediction(d,s)
@time period_prediction(d,s)



loss_fun_plain(((y, u_vals), v_vals), seed, model) = Flux.mse(model([y;H_δ_kh(seed)[3]'],u_vals), v_vals)
loss_fun_plain(d,s,model)
flush(stdout)

# Profile.init(n = 10000000, delay = 0.000001)
# Profile.clear()
# @profile loss_fun_physics_informed(d,s,model)
# Juno.profiler()
# Profile.init(n = 10000000, delay = 0.001)


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

model_filename = "models/trained_model_waves_surface_potential_timestepping_var_depth$(file_postfix).jld2"
if !load_model
    train!(model, loaders, params, loss, opt, n_epochs, loss_train, loss_validation, verbose, loss_fun_plain, model)
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
        loss_test+=loss_fun_plain(d,s,model)/length(loader)
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
    plot_seed = n_u_trajectories + n_u_trajectories_validation + n_u_trajectories_test ÷ 5
    x_plot = xi:(xf-xi)/50:xf
    z_plot = zi:(zf-zi)/50:zf
    y_t0_plot = hcat([[x,z,ti] for x=x_plot for z=z_plot]...)
    y_t1_plot = hcat([[x,z,tf/2] for x=x_plot for z=z_plot]...)
    kh_plot = H_δ_kh(plot_seed)[3][]

    u_vals_plot = u_func(x_locs, plot_seed)
    input_fun_plot = reshape(u_func(y_t0_plot, plot_seed), length(z_plot), length(x_plot))
    v_vals_plot = reshape(v_func(y_t1_plot, plot_seed), length(z_plot), length(x_plot))
    deepo_solution = reshape(model([y_t1_plot;fill(kh_plot,1,size(y_t1_plot,2))], u_vals_plot)[:], length(z_plot), length(x_plot))

    xticks = [xi,(xi+xf)/2,xf]

    h = kh_plot/(2*π/L)
    T = L/(sqrt(g*L/(2*π) * tanh(2*π*h/L)))

    x_plot = x_plot*L
    z_plot = (z_plot.-1)*h
    xticks = xticks*L
    scaled_x_locs = (x_locs .- [0;1]) .* [L;h]

    p1=heatmap(x_plot, z_plot, deepo_solution, reuse = false, title="DeepONet\nprediction", clim=extrema([v_vals_plot;deepo_solution]),xticks=xticks)
    xlabel!("x [m]")
    ylabel!("z [m]")
    title=@sprintf "Error\nMSE %.2e" Flux.mse(deepo_solution, v_vals_plot)
    p2=heatmap(x_plot, z_plot, v_vals_plot-deepo_solution, reuse = false, title=title, yticks=false,xticks=xticks)
    xlabel!("x [m]")
    p3=heatmap(x_plot, z_plot, v_vals_plot, reuse = false, title="Analytical\nsolution", clim=extrema([v_vals_plot;deepo_solution]), yticks=false,xticks=xticks)
    xlabel!("x [m]")
    p = plot(p1, p2, p3, reuse = false, layout = (1,3))
    savefig(p, "plots/small_waves_surface_potential_timestepping_var_depth_example$(file_postfix).pdf")
    display(p)


    title=@sprintf "Error at t=T/2, MSE %.2e" Flux.mse(deepo_solution, v_vals_plot)
    p=plot(x_plot, z_plot, v_vals_plot-deepo_solution, reuse = false, title=title ,xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_surface_potential_timestepping_var_depth_example_3d_error$(file_postfix).pdf")
    display(p)

    p=plot(x_plot, z_plot, deepo_solution, reuse = false, title="DeepONet prediction at t=T/2", clim=extrema([v_vals_plot;deepo_solution]),xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_surface_potential_timestepping_var_depth_example_3d_pred$(file_postfix).pdf")
    display(p)

    p=plot(x_plot, z_plot, v_vals_plot, reuse = false, title="Analytical solution at t=T/2", clim=extrema([v_vals_plot;deepo_solution]), xticks=xticks, st=:surface, right_margin = 4Plots.mm)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_surface_potential_timestepping_var_depth_example_3d_analytical$(file_postfix).pdf")
    display(p)


    p=scatter(scaled_x_locs[1,:],scaled_x_locs[2,:],u_vals_plot,color=:black,label="Sensors")
    plot!(x_plot, z_plot, input_fun_plot, reuse = false, title="Input function (t=0T)", clim=extrema([v_vals_plot;deepo_solution]), xticks=xticks, st=:surface, right_margin = 4Plots.mm, alpha=0.75)
    xlabel!("x [m]")
    ylabel!("z [m]")
    savefig(p, "plots/small_waves_surface_potential_timestepping_var_depth_example_3d_input_fun$(file_postfix).pdf")
    display(p)

    p=plot(loss_train, label="Train", legend=:topright, reuse = false, markershape = :circle, yaxis=:log, title="DeepONet training progress")
    plot!(loss_validation, label="Validation", markershape = :circle)
    xlabel!("Epochs")
    ylabel!("Loss (MSE)")
    savefig(p, "plots/small_waves_surface_potential_timestepping_var_depth_training$(file_postfix).pdf")
    display(p)


    ## Loss vs time step
    khs = zeros(n_u_trajectories_test*n_y_eval)
    losses = zeros(n_u_trajectories_test*n_y_eval)
    if n_freq_gen == 1
        Hs = zeros(n_u_trajectories_test*n_y_eval)
        δs = zeros(n_u_trajectories_test*n_y_eval)
    end
    for (batch_id,(d,s)) in enumerate(loaders.test)
        ((y, u_vals), v_vals) = d
        for i in 1:batch_size
            H,δ,kh = H_δ_kh(s[i])
            if n_freq_gen == 1
                Hs[(batch_id-1)*batch_size + i] = H[]
                δs[(batch_id-1)*batch_size + i] = δ[]
            end
            khs[(batch_id-1)*batch_size + i] = kh[]
            losses[(batch_id-1)*batch_size + i] = loss_fun_plain(((y[:,i:i], (u_vals[1][:,i:i],)), v_vals[1:1,i:i]),s[i],model)
        end
    end

    n_periods = 4
    times = zeros(n_u_trajectories_test*n_y_eval*n_periods)
    losses_time = zeros(n_u_trajectories_test*n_y_eval*n_periods)
    for (batch_id,(d,s)) in enumerate(loaders.test)
        ((y, u_vals), v_vals) = d
        for p in 0:n_periods-1
            for i in 1:batch_size
                times[(batch_id-1)*batch_size + i + p*n_u_trajectories_test*n_y_eval] = y[3,i] + p
                losses_time[(batch_id-1)*batch_size + i + p*n_u_trajectories_test*n_y_eval] = loss_fun_plain(((y[:,i:i]+[0;0;p], (u_vals[1][:,i:i],)), v_vals[1:1,i:i]),s[i],model)
            end
        end
    end

    n_periods_iterative = 16
    times_iterative = zeros(n_u_trajectories_test*n_y_eval*n_periods_iterative)
    losses_time_iterative = zeros(n_u_trajectories_test*n_y_eval*n_periods_iterative)
    for (batch_id,(d,s)) in enumerate(loaders.test)
        ((y, u_vals), v_vals) = d
        u_vals_this_time = repeat(u_vals[1],inner=(1,n_sensors))
        kh = H_δ_kh(s)[3]

        for i in 1:batch_size
            times_iterative[(batch_id-1)*batch_size + i] = y[3,i]
            preds = model([y[:,i:i];kh[i]],(u_vals_this_time[:,i*n_sensors:i*n_sensors],))
            losses_time_iterative[(batch_id-1)*batch_size + i] = Flux.mse(preds, v_vals[:,i:i])
        end

        for p=1:n_periods_iterative-1

            u_vals_this_time = model([repeat(x_locs,1,batch_size);fill(1,1,batch_size*n_sensors);repeat(kh',inner=(1,n_sensors))],(u_vals_this_time,))
            u_vals_this_time = repeat(reshape(u_vals_this_time,n_sensors,batch_size),inner=(1,n_sensors))

            for i in 1:batch_size
                times_iterative[(batch_id-1)*batch_size + i + p*n_u_trajectories_test*n_y_eval] = y[3,i] + p
                preds = model([y[:,i:i];kh[i]],(u_vals_this_time[:,i*n_sensors:i*n_sensors],))
                losses_time_iterative[(batch_id-1)*batch_size + i + p*n_u_trajectories_test*n_y_eval] = Flux.mse(preds, v_vals[:,i:i])
            end
        end
    end



    pyplot_hexbin_times_iterative_inputs = (times_iterative,losses_time_iterative,(0:4:n_periods_iterative, ["0 T", "4 T", "8 T", "12 T", "16 T"]),"Loss vs. time for test set\nRepeated application of DeepONet","Time (unit of wave periods)","Squared error","plots/small_waves_surface_potential_timestepping_var_depth_loss_vs_time_repeated$(file_postfix).pdf")
    pyplot_hexbin_times_inputs = (times,losses_time,(0:1:n_periods, ["0 T", "1 T", "2 T", "3 T", "4 T"]),"Loss vs. time for test set","Time (unit of wave periods)","Squared error","plots/small_waves_surface_potential_timestepping_var_depth_loss_vs_time$(file_postfix).pdf")
    pyplot_hexbin_kh_inputs = (khs,losses,([0.4,1,2,3,4],),"Loss vs. depth scale for test set","Depth scale, kh","Squared error","plots/small_waves_surface_potential_timestepping_var_depth_loss_vs_kh$(file_postfix).pdf")
    if n_freq_gen == 1
        pyplot_hexbin_H_inputs = (Hs,losses,([0.1,0.2,0.3,0.4,0.5],),"Loss vs. amplitude for test set","Amplitude, H [m]","Squared error","plots/small_waves_surface_potential_timestepping_var_depth_loss_vs_H$(file_postfix).pdf")
        pyplot_hexbin_delta_inputs = (δs,losses,([0,π,2π],["0","π","2π"]),"Loss vs. phase for test set","Phase, δ","Squared error","plots/small_waves_surface_potential_timestepping_var_depth_loss_vs_delta$(file_postfix).pdf")

        FileIO.save("hexbin_plot_data$(file_postfix).jld2","pyplot_hexbin_times_inputs",pyplot_hexbin_times_inputs,"pyplot_hexbin_times_iterative_inputs",pyplot_hexbin_times_iterative_inputs,"pyplot_hexbin_kh_inputs",pyplot_hexbin_kh_inputs,"pyplot_hexbin_H_inputs",pyplot_hexbin_H_inputs,"pyplot_hexbin_delta_inputs",pyplot_hexbin_delta_inputs)
    else
        FileIO.save("hexbin_plot_data$(file_postfix).jld2","pyplot_hexbin_times_inputs",pyplot_hexbin_times_inputs,"pyplot_hexbin_times_iterative_inputs",pyplot_hexbin_times_iterative_inputs,"pyplot_hexbin_kh_inputs",pyplot_hexbin_kh_inputs)
    end


end
