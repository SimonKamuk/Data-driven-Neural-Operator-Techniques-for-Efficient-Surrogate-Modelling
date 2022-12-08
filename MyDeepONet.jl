module MyDeepONet

using ProgressBars, Flux, Random, Printf, Distributions, ReverseDiff
export DeepONet, generate_data, train!, evaluate_trunk, evaluate_branch, combine_latent, get_params

struct DeepONet{B,T,B_re,T_re}
    branch::B
    trunk::T
    bias::Vector{Float32}
    const_bias::Bool
    trunk_var_bias::Bool
    branch_p::Vector{Float32}
    trunk_p::Vector{Float32}
    branch_re::B_re
    trunk_re::T_re

    function DeepONet(;branch, trunk, bias=[0.0f0], const_bias=false, trunk_var_bias=false)
        branch_p,branch_re = Flux.destructure(branch)
        trunk_p,trunk_re = Flux.destructure(trunk)

        return new{typeof(branch), typeof(trunk),typeof(branch_re),typeof(trunk_re)}(
            branch,
            trunk,
            bias,
            const_bias,
            trunk_var_bias,
            branch_p,
            trunk_p,
            branch_re,
            trunk_re
        )
    end
end

@Flux.functor DeepONet


function evaluate_branch(m::DeepONet,u_vals,p=nothing)
    if m.branch isa Array
        if p==nothing
            b_funs = m.branch
        else
            b_funs = m.branch_re(p[1])
        end
        b = prod(cat(
                map(1:length(m.branch)) do i
                    b_funs[i](u_vals[i])
                end...,
                dims=3
            ),dims=3)[:,:,1]

    else
        if p==nothing
            b = m.branch(u_vals)
        else
            b = m.branch_re(p[1])(u_vals)
        end
        b = reshape(b,size(b,1),:)
    end
    return b
end
function evaluate_trunk(m::DeepONet,y,p=nothing)
    if p==nothing
        t = m.trunk(y)
    else
        t = m.trunk_re(p[2])(y)
    end
    t = reshape(t,size(t,1),:)
    return t
end
function combine_latent(m::DeepONet,t,b,p=nothing)
    if p==nothing
        bias = m.bias[]
    else
        bias = p[3][]
    end
    if m.trunk_var_bias
        return sum(b .* t[begin:end-1], dims=1) .+ bias .+ t[end]
    else
        return sum(b .* t, dims=1) .+ bias
    end
end

function get_params(m::DeepONet)
    return m.branch_p, m.trunk_p, m.bias
end

function (m::DeepONet)(y,u_vals,p=nothing)
    b = evaluate_branch(m,u_vals,p)
    t = evaluate_trunk(m,y,p)
    return combine_latent(m,t,b,p)
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

    # yspan matrix

    if length(size(yspan)) == 1
        ydim = 1
    else
        ydim = size(yspan, 1)
    end
    n_y_eval_total = prod(n_y_eval)

    if y_locs == nothing
        # if equidistant_y
        #     y_locs = permutedims(cat(
        #                              [
        #                                 repeat(range(start=yspan[i,begin], stop=yspan[i,end], length=n_y_eval[i]), 1, total_trajectories)
        #                              ]...,
        #                              dims=3),[3,1,2])
        # else
        #     y_locs = permutedims(cat(
        #                              [
        #                                  sort(rand(Uniform(yspan[i,begin], yspan[i,end]), n_y_eval[i], total_trajectories), dims=1)
        #                                  for i in 1:ydim
        #                              ]...,
        #                              dims=3),[3,1,2])
        #
        # end
        if equidistant_y
            ranges = [collect(range(start=yspan[i,begin], stop=yspan[i,end], length=n_y_eval[i])) for i in 1:ydim]

            y_locs = ranges[1]
            for r in 2:length(ranges)
                y_locs = cat(
                repeat(y_locs, outer=length(ranges[r])),
                repeat(ranges[r], inner=size(y_locs,1)),
                dims=2
                )
            end

            y_locs = repeat(y_locs', 1, 1, total_trajectories)
        else
            y_locs = permutedims(cat(
                                     [
                                         sort(rand(Uniform(yspan[i,begin], yspan[i,end]), n_y_eval_total, total_trajectories), dims=1)
                                         for i in 1:ydim
                                     ]...,
                                     dims=3),[3,1,2])
        end
    else
        if size(y_locs) == (ydim,n_y_eval_total)
            y_locs = repeat(y_locs, 1, 1, total_trajectories)
        elseif size(y_locs) != (ydim, n_y_eval_total, total_trajectories)
            error("Supplied y_locs with wrong dimensions")
        end
    end


    if u_func isa Array
        u_vals = [zeros((n_sensors[i], n_y_eval_total, total_trajectories)) for i in 1:length(u_func)]
    else
        u_vals = zeros((n_sensors, n_y_eval_total, total_trajectories))
    end
    v_vals = zeros((n_y_eval_total, total_trajectories))
    seeds = zeros(Int, (n_y_eval_total, total_trajectories))

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
        v_vals[:,idx] = v_func(y_locs[:, :, idx], seed)
        seeds[:, idx] .= seed
    end
    print("Initial data generation completed. Finishing up and testing correct ordering now.")

    if u_func isa Array
        u_vals = tuple((reshape(u_vals[i], n_sensors[i], :) for i in 1:length(u_vals))...)
    else
        u_vals = reshape(u_vals, n_sensors, :)
    end
    v_vals = reshape(v_vals, 1, :)
    y_locs = reshape(y_locs, ydim, :)
    seeds = reshape(seeds, :)

    # Split into training, validation, and test sets
    data = ( (y_locs, u_vals), v_vals )

    train_data, validation_data, test_data = Flux.splitobs(data, at=(n_u_trajectories*n_y_eval_total, n_u_trajectories_validation*n_y_eval_total))
    train_seeds, validation_seeds, test_seeds = Flux.splitobs(seeds, at=(n_u_trajectories*n_y_eval_total, n_u_trajectories_validation*n_y_eval_total))

    loader(d) = Flux.DataLoader(Flux.shuffleobs(d), batchsize=batch_size, partial=false, shuffle=true)
    train_loader = loader((train_data,train_seeds)) #Flux.DataLoader(train_data, batchsize=batch_size, partial=false)
    validation_loader = loader((validation_data,validation_seeds)) #Flux.DataLoader(validation_data, batchsize=batch_size, partial=false)
    test_loader = loader((test_data,test_seeds)) #Flux.DataLoader(test_data, batchsize=batch_size, partial=false)
    # train_seeds_loader = loader(train_seeds,copy(train_rng)) #Flux.DataLoader(train_seeds, batchsize=batch_size, partial=false)
    # validation_seeds_loader = loader(validation_seeds,copy(validation_rng)) #Flux.DataLoader(validation_seeds, batchsize=batch_size, partial=false)
    # test_seeds_loader = loader(test_seeds,copy(test_rng)) #Flux.DataLoader(test_seeds, batchsize=batch_size, partial=false)


    # Make sure data is structured as expected
    @assert isempty(intersect(Set(test_seeds), Set(train_seeds), Set(validation_seeds)))
    for loader = (train_loader,validation_loader,test_loader)

        ((y_locs_test, u_vals_test),first_output),first_seeds = first(loader)
        for i in [1,2,3,batch_size ÷ 2,batch_size-3, batch_size-2,batch_size-1,batch_size]
            @assert abs(v_func(y_locs_test[:,i], first_seeds[i])[] - first_output[i]) <= 1e-7
            # @assert v_func(y_locs_test[:,i], first_seeds[i])[] == first_output[i]

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
            validation=validation_loader,
            test=test_loader)

end





function train!(data_loaders, params, loss, opt, n_epochs, loss_train=nothing, loss_validation=nothing)
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss

    # lowest_loss = Inf
    # lowest_loss_epoch = 0
    # last_reduce_epoch = -Inf

    if loss_train == nothing
        loss_train = fill(NaN,n_epochs)
    end
    if loss_validation == nothing
        loss_validation = fill(NaN,n_epochs)
    end
    training_progress = ProgressBar(1:n_epochs)
    set_description(training_progress, "Training DeepONet:")
    for e in training_progress
        loss_t=0
        loss_v=0
        for (d,s) in data_loaders.train
            if params isa Tuple
                gs = ReverseDiff.gradient(params) do p...
                    training_loss = loss(d...,p...)
                    return training_loss
                end
            else
                gs = Flux.gradient(params) do
                    training_loss = loss(d...)
                    return training_loss
                end
            end
            # Update gradient and save training loss
            Flux.update!(opt, params, gs)
            loss_t+=training_loss
        end
        # Make sure we get the mean loss
        loss_train[e]=loss_t/length(data_loaders.train)


        # Compute mean validation loss
        for (d,s) in data_loaders.validation
            loss_v+=loss(d...)
        end
        loss_validation[e]=loss_v/length(data_loaders.validation)
        set_multiline_postfix(training_progress, @sprintf("Training loss: %.3e\nValidation loss: %.3e", loss_train[e], loss_validation[e]))
    end

    return loss_train, loss_validation
end





end
