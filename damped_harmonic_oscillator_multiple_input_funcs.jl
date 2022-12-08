using ModelingToolkit, DifferentialEquations, Plots, GaussianRandomFields, Interpolations, Random, Flux, IterTools, GraphNeuralNetworks, Statistics, LinearAlgebra, Graphs, SparseArrays, Juno, ForwardDiff, Printf, ProgressBars, Distributions


include("MyDeepONet.jl")
using .MyDeepONet


yspan = [0 3]
v0 = [1,1] # initial value of solution at y=0
# Define problem to be solved: dv/dy = u(y)

function f(v, p, y)
    u,ζ,k,m = p
    dv = [v[2], -2*ζ*sqrt(k/m)*v[2] .- k/m*v[1] .+ u(y)/m]
    return dv
end

l = 0.2 * sqrt(2)
n_dims = 1
n_sensors = 100
nn_width = 50
latent_size = 50
activation_function = relu
n_grf_generate_points = 1000
n_u_trajectories = 200
n_u_trajectories_test = 1000
n_u_trajectories_validation = 200
batch_size = 100
n_y_eval = 1000
xi = yspan[1]
xf = yspan[2]
recompute_data = true
Random.seed!(0)
flux_ini = Flux.glorot_uniform(MersenneTwister(rand(Int64)))

x_locs = range(start=xi, stop=xf, length=n_sensors) # Sensor locations (input function evaluation points)



## Define functions

if !(@isdefined grf) | recompute_data
    println("Generating gaussian random field")
    kernel = Gaussian(l, σ=1, p=2)
    cov = CovarianceFunction(n_dims, kernel)
    grf_generate_point_locs = range(start=xi, stop=xf, length=n_grf_generate_points)
    grf = GaussianRandomField(cov, Spectral(), grf_generate_point_locs)
end

function get_mass(seed)
    rng = MersenneTwister(seed)
    m = 0.0
    while m>=0.25 || m<=0.05
        m = randn(rng) * 0.05 + 0.15
    end
    # while m<=0
    #     m = randn(rng) * 0.05 + 0.15
    # end

    # m=0.15
    return m, rng
end


function get_u_grf(seed)
    # Define input function
    m,rng = get_mass(seed)

    ζ,k = 1,1

    interp = interpolate(
        (grf_generate_point_locs,),
        GaussianRandomFields.sample(grf,xi=randn(rng, randdim(grf))),
        Gridded(Interpolations.Linear()))
    return interp
end


function get_u_sinusoidal(seed)
    # Define input function
    m,rng = get_mass(seed)
    ζ,k = 1,1
    F0=rand(rng, Uniform(0,6))
    ω=rand(rng, Uniform(0,10))
    return x->F0*sin.(ω*x)
end



function v_func_analytical_sinusoidal(y,seed;m_in=nothing)
    m,rng = get_mass(seed)
    if m_in != nothing
        m=m_in
    end
    ζ,k = 1,1
    F0=rand(rng, Uniform(0,6))
    ω=rand(rng, Uniform(0,10))

    ω₀= sqrt(k/m)

    φ = atan(2*ω*ω₀*ζ/(ω^2-ω₀^2))
    φ += (φ>0)*π
    Z = sqrt((2*ω₀*ζ)^2 + ((ω₀^2-ω^2)/ω)^2)
    steady_state = F0*sin.(ω*y.+φ) / (m*Z*ω)

    c_1=-exp(sqrt(k)*yspan[1]/sqrt(m))*(-F0*(sqrt(k)*yspan[1] - sqrt(m))*sin(ω*yspan[1] + φ) + (-sqrt(m)*F0*yspan[1]*cos(ω*yspan[1] + φ) + ((v0[2]*yspan[1] - v0[1])*m^(3/2) + sqrt(k)*m*v0[1]*yspan[1])*Z)*ω)/(m^(3/2)*Z*ω)
    c_2=exp(sqrt(k)*yspan[1]/sqrt(m))*(-F0*cos(ω*yspan[1] + φ)*ω*sqrt(m) - F0*sin(ω*yspan[1] + φ)*sqrt(k) + Z*ω*(sqrt(k)*m*v0[1] + v0[2]*m^(3/2)))/(m^(3/2)*Z*ω)

    if ζ==1
        transient = (c_1.+c_2*y).*exp.(-sqrt(k/m)*y)
    end
    return steady_state+transient
end

get_u = get_u_grf

function v_func(y, seed)
    # Solve problem with numerical ode solver (4th order Runge-Kutta) and
    # evaluate solution at points y

    u=get_u(seed)
    m,rng = get_mass(seed)
    ζ,k = 1,1

    p = (u,ζ,k,m)

    prob = ODEProblem(f, v0, yspan, p)
    v_values = solve(prob, Tsit5(), saveat=y).u
    # Return only the solution, not its derivative
    return [v[1] for v in v_values]
end

u_func = [(x_locs, seed) -> get_u(seed)(x_locs), (x_locs, seed) -> [get_mass(seed)[1]]]
x_locs_tuple = [x_locs, []]
n_sensors_tuple = [n_sensors, 1]

## Generate data
if !(@isdefined loaders) | recompute_data
    loaders = generate_data(x_locs_tuple, yspan, u_func, v_func, n_sensors_tuple, n_u_trajectories, n_u_trajectories_validation, n_u_trajectories_test, n_y_eval, batch_size, equidistant_y=false)
end

## Define layers

branch = [Chain(Dense(n_sensors, nn_width, activation_function, init=flux_ini),
                Dense(nn_width, latent_size, init=flux_ini)),
          Chain(Dense(1, nn_width, activation_function, init=flux_ini),
                Dense(nn_width, nn_width, activation_function, init=flux_ini),
                Dense(nn_width, latent_size, init=flux_ini))]

# branch = [Chain(Dense(n_sensors, latent_size, init=flux_ini)),
#           Chain(x->ones(latent_size,size(x)[2:end]...))]



trunk = Chain(
    Dense(1, nn_width, activation_function, init=flux_ini),
    Dense(nn_width, nn_width, activation_function, init=flux_ini),
    Dense(nn_width, latent_size, activation_function, init=flux_ini)
)

# Define model
model = DeepONet(trunk=trunk, branch=branch, const_bias=true)
loss((y, u_vals), v_y_true) = Flux.mse(model(y,u_vals), v_y_true)
params = Flux.params(model)

loss(first(loaders.train)[1]...)

## Training loop
opt = NAdam()
# opt = Adam()

n_epochs = 200

loss_train, loss_validation = train!(loaders, params, loss, opt, n_epochs)

# To be used only after final model is selected
function get_loss_test()
    loss_test = 0
    for (d,s) in loaders.test
        loss_test+=loss(d...)/length(loaders.test)
    end
    return loss_test
end
loss_test = get_loss_test()
println(@sprintf "Test loss: %.3e" loss_test)






## Plotting
plot_seed = n_u_trajectories + n_u_trajectories_validation + n_u_trajectories_test÷4
u_vals_plot = [u_func[i](x_locs_tuple[i], plot_seed) for i in 1:length(u_func)]
v_vals_plot = v_func(x_locs, plot_seed)
deepo_solution = model(reshape(x_locs,1,:), u_vals_plot)[:]
title = @sprintf "Example input/output. MSE %.2e. Mass %.3f" Flux.mse(deepo_solution, v_vals_plot) u_vals_plot[2][]
p=plot(x_locs, u_vals_plot[1], label="Input function from test set", reuse = false, title=title)
plot!(x_locs, v_vals_plot, label="Numerical solution")
plot!(x_locs, deepo_solution, label="DeepONet output")
xlabel!("y")
ylabel!("Function value")
savefig(p, "plots/harmonic_oscillator_test_function_var_mass.pdf")
display(p)





m = 0.1
y_vals_plot = yspan[1]:0.001:yspan[2]
u_vals_plot = get_u_sinusoidal(plot_seed)(x_locs)
v_vals_plot = v_func_analytical_sinusoidal(y_vals_plot, plot_seed;m_in=m)
deepo_solution = model(reshape(y_vals_plot,1,:), [u_vals_plot, [m]])[:]
title = @sprintf "Harmonic oscillator. MSE %.2e. Mass %.2f" Flux.mse(deepo_solution, v_vals_plot) m
p=plot(x_locs, u_vals_plot, label="Input function: F₀sin(ωy)", reuse = false, title=title)
plot!(y_vals_plot, v_vals_plot, label="Analytical solution")
plot!(y_vals_plot, deepo_solution, label="DeepONet output")
xlabel!("y")
ylabel!("Function value")
savefig(p, "plots/harmonic_oscillator_analytical_var_mass_$m.pdf")
display(p)


m = 0.15
y_vals_plot = yspan[1]:0.001:yspan[2]
u_vals_plot = get_u_sinusoidal(plot_seed)(x_locs)
v_vals_plot = v_func_analytical_sinusoidal(y_vals_plot, plot_seed;m_in=m)
deepo_solution = model(reshape(y_vals_plot,1,:), [u_vals_plot, [m]])[:]
title = @sprintf "Harmonic oscillator. MSE %.2e. Mass %.2f" Flux.mse(deepo_solution, v_vals_plot) m
p=plot(x_locs, u_vals_plot, label="Input function: F₀sin(ωy)", reuse = false, title=title)
plot!(y_vals_plot, v_vals_plot, label="Analytical solution")
plot!(y_vals_plot, deepo_solution, label="DeepONet output")
xlabel!("y")
ylabel!("Function value")
savefig(p, "plots/harmonic_oscillator_analytical_var_mass_$m.pdf")
display(p)

m = 0.2
y_vals_plot = yspan[1]:0.001:yspan[2]
u_vals_plot = get_u_sinusoidal(plot_seed)(x_locs)
v_vals_plot = v_func_analytical_sinusoidal(y_vals_plot, plot_seed;m_in=m)
deepo_solution = model(reshape(y_vals_plot,1,:), [u_vals_plot, [m]])[:]
title = @sprintf "Harmonic oscillator. MSE %.2e. Mass %.2f" Flux.mse(deepo_solution, v_vals_plot) m
p=plot(x_locs, u_vals_plot, label="Input function: F₀sin(ωy)", reuse = false, title=title)
plot!(y_vals_plot, v_vals_plot, label="Analytical solution")
plot!(y_vals_plot, deepo_solution, label="DeepONet output")
xlabel!("y")
ylabel!("Function value")
savefig(p, "plots/harmonic_oscillator_analytical_var_mass_$m.pdf")
display(p)

m = 0.25
y_vals_plot = yspan[1]:0.001:yspan[2]
u_vals_plot = get_u_sinusoidal(plot_seed)(x_locs)
v_vals_plot = v_func_analytical_sinusoidal(y_vals_plot, plot_seed;m_in=m)
deepo_solution = model(reshape(y_vals_plot,1,:), [u_vals_plot, [m]])[:]
title = @sprintf "Harmonic oscillator. MSE %.2e. Mass %.2f" Flux.mse(deepo_solution, v_vals_plot) m
p=plot(x_locs, u_vals_plot, label="Input function: F₀sin(ωy)", reuse = false, title=title)
plot!(y_vals_plot, v_vals_plot, label="Analytical solution")
plot!(y_vals_plot, deepo_solution, label="DeepONet output")
xlabel!("y")
ylabel!("Function value")
savefig(p, "plots/harmonic_oscillator_analytical_var_mass_$m.pdf")
display(p)





p=plot(loss_train, label="Train", legend=:topright, reuse = false, markershape = :circle, yaxis=:log, title="DeepONet training progress")
plot!(loss_validation, label="Validation", markershape = :circle)
xlabel!("Epochs")
ylabel!("Loss (MSE)")
savefig(p, "plots/harmonic_oscillator_training_var_mass.pdf")
display(p)
