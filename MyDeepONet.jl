module MyDeepONet


import Base.@kwdef
using ProgressBars, Flux, Random, Printf
export DeepONet, generate_data, train!

@kwdef struct DeepONet
    branch
    trunk
    bias = [0.0]
    const_bias::Bool = false
    trunk_var_bias::Bool = false
end


function (m::DeepONet)(y,u_vals)
    if m.trunk_var_bias
        b = m.branch(u_vals)
        t = m.trunk(y)
        return sum(b .* t[begin:end-1], dims=1) .+ m.bias[] .+ t[end]
    else
        b = m.branch(u_vals)
        t = m.trunk(y)
        return sum(b .* t, dims=1) .+ m.bias[]
    end
end

function Flux.trainable(m::DeepONet)
    if m.const_bias
        return (m.branch, m.trunk, m.bias)
    else
        return (m.branch, m.trunk)
    end
end


function generate_data(x_locs, y_locs, u_func, v_func, n_sensors, n_u_trajectories, n_u_trajectories_test, n_u_trajectories_validation, n_y_eval, batch_size)


    u_vals = zeros((n_sensors, n_u_trajectories+n_u_trajectories_test+n_u_trajectories_validation, n_y_eval))
    v_vals = zeros((n_u_trajectories+n_u_trajectories_test+n_u_trajectories_validation, n_y_eval))
    seeds = zeros(Int, (n_u_trajectories+n_u_trajectories_test+n_u_trajectories_validation, n_y_eval))

    seeds_progress = ProgressBar(1:n_u_trajectories+n_u_trajectories_test+n_u_trajectories_validation)
    set_description(seeds_progress, "Generating data:")
    for seed in seeds_progress
        u_vals[:, seed, :] .= u_func(x_locs, seed)
        v_vals[seed, :] = v_func(y_locs, seed)
        seeds[seed, :] .= seed
    end
    print("Initial data generation completed. Finishing up and testing correct ordering now.")

    y_idx = repeat(reshape(Array(1:n_y_eval), 1, n_y_eval), n_u_trajectories+n_u_trajectories_test+n_u_trajectories_validation, 1)
    u_vals = reshape(permutedims(u_vals,[1,3,2]), n_sensors, :)
    v_vals = reshape(permutedims(v_vals,[2,1]), 1, :)
    y_idx = reshape(permutedims(y_idx,[2,1]), 1, :)
    seeds = reshape(permutedims(seeds,[2,1]), :)



    # Split into training, validation, and test sets
    data = ((y_locs[y_idx], u_vals), v_vals)
    train_data, validation_data, test_data = Flux.splitobs(data, at=(n_u_trajectories*n_y_eval, n_u_trajectories_validation*n_y_eval), shuffle=false)
    train_seeds, validation_seeds, test_seeds = Flux.splitobs(seeds, at=(n_u_trajectories*n_y_eval, n_u_trajectories_validation*n_y_eval), shuffle=false)

    train_loader = Flux.DataLoader(train_data, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
    validation_loader = Flux.DataLoader(validation_data, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
    test_loader = Flux.DataLoader(test_data, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
    train_seeds_loader = Flux.DataLoader(train_seeds, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
    validation_seeds_loader = Flux.DataLoader(validation_seeds, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)
    test_seeds_loader = Flux.DataLoader(test_seeds, batchsize=batch_size, shuffle=true, rng=MersenneTwister(0), partial=false)


    # Make sure data is structured as expected
    for (data_loader, seeds_loader) in ((train_loader,train_seeds_loader),
                                        (validation_loader,validation_seeds_loader),
                                        (test_loader,test_seeds_loader),)

        first_input,first_output = first(data_loader)
        first_seeds = first(seeds_loader)
        y_locs_test, u_vals_test = first_input
        for i in [1,2,3,batch_size ÷ 2,batch_size-3, batch_size-2,batch_size-1,batch_size]
            @assert v_func(y_locs_test[:, i], first_seeds[i])[] == first_output[i]
            @assert u_func(x_locs, first_seeds[i]) == u_vals_test[:,i]
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





function train!(data_loaders, params, loss, opt, n_epochs; lr_factor = 0.2, patience = 5, threshold_factor = 0.9, cooldown=3)
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

        # Reduce learning rate on plateau
        if loss_train[e] < lowest_loss && (e-last_reduce_epoch>cooldown)
            lowest_loss = loss_train[e]
            lowest_loss_epoch = e
        elseif (e-lowest_loss_epoch >= patience) && (loss_train[e] >= threshold_factor*lowest_loss)
            opt.eta *= lr_factor
            last_reduce_epoch = e
        end


        # Compute mean validation loss
        for d in data_loaders.validation
            loss_validation[e]+=loss(d...)
        end
        loss_validation[e]/=length(data_loaders.validation)
        set_multiline_postfix(training_progress, @sprintf("Learning rate: %.3e\nTraining loss: %.3e\nValidation loss: %.3e", opt.eta, loss_train[e], loss_validation[e]))
        # set_multiline_postfix(training_progress, @sprintf("Training loss: %.3e\nValidation loss: %.3e", loss_train[e], loss_validation[e]))
    end

    return loss_train, loss_validation
end





end
