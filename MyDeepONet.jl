module MyDeepONet


import Base.@kwdef
using ProgressBars, Flux, Random, Printf, Distributions
export DeepONet, generate_data, train!

@kwdef struct DeepONet
    branch
    trunk
    bias = [0.0]
    const_bias::Bool = false
    trunk_var_bias::Bool = false
end


function (m::DeepONet)(y,u_vals)
    if m.branch isa Array
        b = prod(cat(
                [m.branch[i](u_vals[i]) for i in 1:length(m.branch)]...,
                dims=3
            ),dims=3)[:,:,1]

    else
        b = m.branch(u_vals)
        b = reshape(b,size(b,1),:)
    end
    t = m.trunk(y)
    t = reshape(t,size(t,1),:)

    if m.trunk_var_bias
        return sum(b .* t[begin:end-1], dims=1) .+ m.bias[] .+ t[end]
    else
        return sum(b .* t, dims=1) .+ m.bias[]
    end
end

function Flux.trainable(m::DeepONet)
    if m.branch isa Array
        if m.const_bias
            return (m.branch..., m.trunk, m.bias)
        else
            return (m.branch..., m.trunk)
        end
    else
        if m.const_bias
            return (m.branch, m.trunk, m.bias)
        else
            return (m.branch, m.trunk)
        end
    end
end


function generate_data(x_locs, yspan, u_func, v_func, n_sensors, n_u_trajectories, n_u_trajectories_validation, n_u_trajectories_test, n_y_eval, batch_size; equidistant_y=false, y_locs=nothing)
    total_trajectories = n_u_trajectories+n_u_trajectories_test+n_u_trajectories_validation

    if y_locs == nothing
        if equidistant_y
            y_locs = repeat(range(start=yspan[begin], stop=yspan[end], length=n_y_eval), 1, total_trajectories)
        else
            y_locs = rand(Uniform(yspan[begin], yspan[end]), n_y_eval, total_trajectories)
            sort!(y_locs, dims=1)
        end
    else
        if length(size(y_locs)) == 1
            y_locs = repeat(y_locs, 1, total_trajectories)
        end
    end

    if u_func isa Array
        u_vals = [zeros((n_sensors[i], n_y_eval, total_trajectories)) for i in 1:length(u_func)]
    else
        u_vals = zeros((n_sensors, n_y_eval, total_trajectories))
    end
    v_vals = zeros((n_y_eval, total_trajectories))
    seeds = zeros(Int, (n_y_eval, total_trajectories))

    seeds_progress = ProgressBar(Flux.shuffleobs(MersenneTwister(abs(rand(Int64))),1:total_trajectories))
    set_description(seeds_progress, "Generating data:")

    for (idx,seed) in enumerate(seeds_progress)
        if u_func isa Array
            for u_idx in 1:length(u_func)
                u_vals[u_idx][:, :, idx] .= u_func[u_idx](x_locs[u_idx], seed)
            end
        else
            u_vals[:, :, idx] .= u_func(x_locs, seed)
        end
        v_vals[:,idx] = v_func(y_locs[:, idx], seed)
        seeds[:, idx] .= seed
    end
    print("Initial data generation completed. Finishing up and testing correct ordering now.")

    if u_func isa Array
        u_vals = tuple((reshape(u_vals[i], n_sensors[i], :) for i in 1:length(u_vals))...)
    else
        u_vals = reshape(u_vals, n_sensors, :)
    end
    v_vals = reshape(v_vals, 1, :)
    y_locs = reshape(y_locs, 1, :)
    seeds = reshape(seeds, :)

    # Split into training, validation, and test sets
    data = ( (y_locs, u_vals), v_vals )

    train_data, validation_data, test_data = Flux.splitobs(data, at=(n_u_trajectories*n_y_eval, n_u_trajectories_validation*n_y_eval))
    train_seeds, validation_seeds, test_seeds = Flux.splitobs(seeds, at=(n_u_trajectories*n_y_eval, n_u_trajectories_validation*n_y_eval))

    train_rng = MersenneTwister(abs(rand(Int64)))
    test_rng = MersenneTwister(abs(rand(Int64)))
    validation_rng = MersenneTwister(abs(rand(Int64)))

    loader(d,rng) = Flux.DataLoader(Flux.shuffleobs(rng,d), batchsize=batch_size, partial=false)
    train_loader = loader(train_data,copy(train_rng)) #Flux.DataLoader(train_data, batchsize=batch_size, partial=false)
    validation_loader = loader(validation_data,copy(validation_rng)) #Flux.DataLoader(validation_data, batchsize=batch_size, partial=false)
    test_loader = loader(test_data,copy(test_rng)) #Flux.DataLoader(test_data, batchsize=batch_size, partial=false)
    train_seeds_loader = loader(train_seeds,copy(train_rng)) #Flux.DataLoader(train_seeds, batchsize=batch_size, partial=false)
    validation_seeds_loader = loader(validation_seeds,copy(validation_rng)) #Flux.DataLoader(validation_seeds, batchsize=batch_size, partial=false)
    test_seeds_loader = loader(test_seeds,copy(test_rng)) #Flux.DataLoader(test_seeds, batchsize=batch_size, partial=false)
    

    # Make sure data is structured as expected
    @assert isempty(intersect(Set(test_seeds), Set(train_seeds), Set(validation_seeds_loader)))
    for (data_loader, seeds_loader) = ((train_loader,train_seeds_loader),
                                        (validation_loader,validation_seeds_loader),
                                        (test_loader,test_seeds_loader),)

        first_input,first_output = first(data_loader)
        first_seeds = first(seeds_loader)
        y_locs_test, u_vals_test = first_input
        for i in [1,2,3,batch_size ÷ 2,batch_size-3, batch_size-2,batch_size-1,batch_size]
            @assert v_func(y_locs_test[:,i], first_seeds[i])[] == first_output[i]

            if u_func isa Array
                for u_idx in 1:length(u_func)
                    @assert u_func[u_idx](x_locs[u_idx], first_seeds[i]) == u_vals_test[u_idx][:, i]
                end
            else
                @assert u_func(x_locs, first_seeds[i]) == u_vals_test[:,i]
            end
        end
    end



    println("\e[2K","\e[1G","Data generation finished")

    return (train=train_loader,
            train_seeds=train_seeds_loader,
            validation=validation_loader,
            validation_seeds=validation_seeds_loader,
            test=test_loader,
            test_seeds=test_seeds_loader,)

end





function train!(data_loaders, params, loss, opt, n_epochs)
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss

    lowest_loss = Inf
    lowest_loss_epoch = 0
    last_reduce_epoch = -Inf

    loss_train = zeros(n_epochs)
    loss_validation = zeros(n_epochs)
    training_progress = ProgressBar(1:n_epochs)
    set_description(training_progress, "Training DeepONet:")
    for e in training_progress
        for d in data_loaders.train
            gs = Flux.gradient(params) do
                # Code inserted here will be differentiated
                training_loss = loss(d...)
                return training_loss
            end

            # Update gradient and save training loss
            Flux.update!(opt, params, gs)
            loss_train[e]+=training_loss
        end
        # Make sure we get the mean loss
        loss_train[e]/=length(data_loaders.train)


        # Compute mean validation loss
        for d in data_loaders.validation
            loss_validation[e]+=loss(d...)
        end
        loss_validation[e]/=length(data_loaders.validation)
        set_multiline_postfix(training_progress, @sprintf("Training loss: %.3e\nValidation loss: %.3e", loss_train[e], loss_validation[e]))
    end

    return loss_train, loss_validation
end





end
