using Flux, Plots, StatsPlots

input_filename = "output/deeponet_param_search_15136559_"
ext = ".out"
min_loss = (Inf,-1)


params = [
(n_sensors,width,latent_width,activation_function,depth,physics_weight)
for n_sensors in [50,100,150]
for width in [50,75,100]
for latent_width in [50,75,100]
for activation_function in [softplus,tanh,sigmoid]
for depth in [3,4,5]
for physics_weight in [0.1,0.5,1.0]
]

n_params = length(params[1])
# (n_sensors,nn_width,latent_size,activation_function,branch_depth,trunk_depth,physics_weight) = params[jobindex]
loss_values = [Dict() for _ in 1:n_params]
    # Dict(50=>[],100=>[],150=>[]),
    # Dict(50=>[],75=>[],100=>[]),
    # Dict(50=>[],75=>[],100=>[]),
    # Dict(softplus=>[],tanh=>[],sigmoid=>[]),
    # Dict(3=>[],4=>[],5=>[]),
    # Dict(0.1=>[],0.5=>[],1.0=>[]),

param_names = [
    "Number of sensors",
    "Network width",
    "Latent space width",
    "Activation function",
    "Network depth",
    "Physics weight",
]

flush(stdout)

for i = 1:length(params)
    global s, min_loss
    file = input_filename * string(i) * ext

    s = open(file) do f
        read(f, String)
    end

    if !occursin("Successfully completed.",s)
        println("Not successful: $i with params $(params[i])")
        continue
    end
    val_loss_begin = findfirst("Mean of last 10 validation errors:\n",s)[end]+1
    val_loss_end = findfirst("\n",s[val_loss_begin+1:end])[begin]
    loss = parse(Float64, s[val_loss_begin:val_loss_begin+val_loss_end-1])

    # println(loss)

    if min_loss[1]>loss
        min_loss = (loss,i)
    end

    for param_type_id in 1:length(loss_values)
        param_value = params[i][param_type_id]
        if !haskey(loss_values[param_type_id], param_value)
            loss_values[param_type_id][param_value] = []
        end
        append!(loss_values[param_type_id][param_value], loss)
    end
end

println("\nLowest mean of last 10 validation losses: $(min_loss[1])\nat index $(min_loss[2]) with params $(params[min_loss[2]])")

##

for param_type_id in 1:length(loss_values)
    title = param_names[param_type_id]
    param_loss = Vector.(values(loss_values[param_type_id]))
    x_labels = [keys(loss_values[param_type_id])...]

    try
        global sort_idx = sortperm(x_labels)
    catch
        global sort_idx = 1:length(x_labels)
    end

    x_labels = string.(x_labels[sort_idx])
    param_loss = param_loss[sort_idx]

    for label_id in 1:length(x_labels)
        n_rep = length(x_labels) - label_id
        x_labels[label_id] = repeat(" ", n_rep*10) * x_labels[label_id] * repeat(" ", n_rep*10)
    end
    x_labels = repeat(x_labels, inner=length(param_loss[1]))
    param_loss = vcat(param_loss...)
    p=StatsPlots.boxplot(x_labels, param_loss, legend=false, yscale=:log10, whisker_range=Inf)
    title!(title)
    ylabel!("Loss, mean of last 10 epochs")
    savefig(p, "plots/conv_diff_param_search_$(lowercase(replace(title," "=>"_"))).pdf")
    display(p)

end
