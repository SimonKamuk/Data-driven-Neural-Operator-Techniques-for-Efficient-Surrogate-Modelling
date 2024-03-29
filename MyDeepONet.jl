module MyDeepONet

using ProgressBars, Flux, Random, Printf, Distributions, ReverseDiff, Optimisers
export DeepONet, generate_data, train!, evaluate_trunk, evaluate_branch, combine_latent, get_params, generate_y_locs

struct DeepONet{B,T,B_re,T_re,branch_float_type,trunk_float_type,bias_float_type}
    branch::B
    trunk::T
    bias::Vector{ET} where ET<:bias_float_type
    const_bias_trainable::Bool
    trunk_var_bias::Bool
    branch_p::Vector{ET} where ET<:branch_float_type
    trunk_p::Vector{ET} where ET<:trunk_float_type
    branch_re::B_re
    trunk_re::T_re

    function DeepONet(;branch, trunk, bias=[0.0f0], const_bias_trainable=false, trunk_var_bias=false)
        branch_p,branch_re = Flux.destructure(branch)
        trunk_p,trunk_re = Flux.destructure(trunk)

        if !(branch isa Array)
            branch = [branch]
        end
        if !(trunk isa Array)
            trunk = [trunk]
        end

        return new{typeof(branch), typeof(trunk),typeof(branch_re),typeof(trunk_re),eltype(branch_p),eltype(trunk_p),eltype(bias)}(
            branch,
            trunk,
            bias,
            const_bias_trainable,
            trunk_var_bias,
            branch_p,
            trunk_p,
            branch_re,
            trunk_re
        )
    end
end

@Flux.functor DeepONet


function evaluate_branch(m::DeepONet,u_vals,p)
    b_funs = m.branch_re(p[1])
    if !(u_vals isa Tuple)
        u_vals = (u_vals,)
    end
    b = prod(cat(
            map(1:length(m.branch)) do i
                b_funs[i](u_vals[i])
            end...,
            dims=3
        ),dims=3)[:,:,1]
    return b
end
function evaluate_branch(m::DeepONet,u_vals)
    b_funs = m.branch
    if !(u_vals isa Tuple)
        u_vals = (u_vals,)
    end
    b = prod(cat(
            map(1:length(m.branch)) do i
                b_funs[i](u_vals[i])
            end...,
            dims=3
        ),dims=3)[:,:,1]
    return b
end
function evaluate_trunk(m::DeepONet,y,p)
    t_vals = [t_fun(y) for t_fun in m.trunk_re(p[2])]
    return cat([reshape(t_val,:,size(y,2)) for t_val in t_vals]..., dims=3)
end
function evaluate_trunk(m::DeepONet,y)
    t_vals = [t_fun(y) for t_fun in m.trunk]
    return cat([reshape(t_val,:,size(y,2)) for t_val in t_vals]..., dims=3)
end
function combine_latent(m::DeepONet,t,b,p)

    bias = p[3][]

    if m.trunk_var_bias
        return (sum(b .* t[begin:end-1,:,:], dims=1)[1,:,:] .+ bias .+ t[end,:,:])'
    else
        return (sum(b .* t, dims=1)[1,:,:] .+ bias)'
    end
end
function combine_latent(m::DeepONet,t,b)
    bias = m.bias[]

    if m.trunk_var_bias
        return (sum(b .* t[begin:end-1,:,:], dims=1)[1,:,:] .+ bias .+ t[end,:,:])'
    else
        return (sum(b .* t, dims=1)[1,:,:] .+ bias)'
    end
end
function restructure_update(m::DeepONet)
    branch_params = Flux.params(m.branch)
    branch_re_params = Flux.params(m.branch_re(m.branch_p))
    trunk_params = Flux.params(m.trunk)
    trunk_re_params = Flux.params(m.trunk_re(m.trunk_p))

    for i=1:length(branch_params)
        branch_params[i] .= branch_re_params[i]
    end
    for i=1:length(trunk_params)
        trunk_params[i] .= trunk_re_params[i]
    end
end

function get_params(m::DeepONet)
    if m.const_bias_trainable
        return [m.branch_p, m.trunk_p, m.bias]
    else
        return [m.branch_p, m.trunk_p]
    end
end


function (m::DeepONet)(y,u_vals,p)
    b = evaluate_branch(m,u_vals,p)
    t = evaluate_trunk(m,y,p)
    return combine_latent(m,t,b,p)
end
function (m::DeepONet)(y,u_vals)
    b = evaluate_branch(m,u_vals)
    t = evaluate_trunk(m,y)
    return combine_latent(m,t,b)
end


function Flux.trainable(m::DeepONet)
    if m.branch isa Array
        if m.const_bias_trainable
            return (m.branch..., m.trunk, m.bias)
        else
            return (m.branch..., m.trunk)
        end
    else
        if m.const_bias_trainable
            return (m.branch, m.trunk, m.bias)
        else
            return (m.branch, m.trunk)
        end
    end
end

function generate_y_locs(yspan, n_y_eval, total_trajectories, equidistant_y)
    if length(size(yspan)) == 1
        ydim = 1
    else
        ydim = size(yspan, 1)
    end
    n_y_eval_total = prod(n_y_eval)

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
                                     rand(Uniform(yspan[i,begin], yspan[i,end]), n_y_eval_total, total_trajectories)
                                     # sort(rand(Uniform(yspan[i,begin], yspan[i,end]), n_y_eval_total, total_trajectories), dims=1)
                                     for i in 1:ydim
                                 ]...,
                                 dims=3),[3,1,2])
    end
end

function generate_data(x_locs, yspan, u_func, v_func, n_sensors, n_u_trajectories, n_u_trajectories_validation, n_u_trajectories_test, n_y_eval, batch_size; equidistant_y=false, y_locs=nothing, vdim=1)
    total_trajectories = n_u_trajectories+n_u_trajectories_test+n_u_trajectories_validation

    # yspan matrix

    if length(size(yspan)) == 1
        ydim = 1
    else
        ydim = size(yspan, 1)
    end
    n_y_eval_total = prod(n_y_eval)

    if y_locs == nothing
        y_locs = generate_y_locs(yspan, n_y_eval, total_trajectories, equidistant_y)
    else
        if size(y_locs) == (ydim,n_y_eval_total)
            y_locs = repeat(y_locs, 1, 1, total_trajectories)
        elseif size(y_locs) != (ydim, n_y_eval_total, total_trajectories)
            error("Supplied y_locs with wrong dimensions")
        end
    end


    # if !(n_sensors isa Array)
    #     n_sensors = [n_sensors]
    # end
    if !(x_locs isa Tuple)
        if length(size(x_locs)) == 1
            n_sensors = (size(x_locs,1),)
        else
            n_sensors = (size(x_locs,2),)
        end
        udim = 1
    else
        n_sensors = map(x_locs) do x
            if length(size(x)) == 1
                return size(x,1)
            else
                return size(x,2)
            end
        end

        udim = length(x_locs)
    end

    # if u_func isa Array
    u_vals = tuple((zeros((n_sensors[i], n_y_eval_total, total_trajectories)) for i in 1:udim)...)
    # else
    #     u_vals = zeros((n_sensors, n_y_eval_total, total_trajectories))
    # end

    v_vals = zeros((vdim, n_y_eval_total, total_trajectories))
    seeds = zeros(Int, (n_y_eval_total, total_trajectories))

    seeds_progress = ProgressBar(Flux.shuffleobs(MersenneTwister(abs(rand(Int64))),1:total_trajectories))
    set_description(seeds_progress, "Generating data:")

    for (idx,seed) in enumerate(seeds_progress)
        # if u_func isa Array
            u_vals_this_seed = u_func(x_locs, seed)
            if !(u_vals_this_seed isa Tuple)
                u_vals_this_seed = (u_vals_this_seed,)
            end

            for u_idx in 1:udim
                u_vals[u_idx][:, :, idx] .= u_vals_this_seed[u_idx]
            end

        # else
        #     u_vals[:, :, idx] .= u_func(x_locs, seed)
        # end
        v_vals[:, :,idx] = v_func(y_locs[:, :, idx], seed)
        # println(y_locs[:, :, idx], seed)
        # println(v_vals[:, :,idx])
        # println("\n")
        seeds[:, idx] .= seed
    end
    print("Initial data generation completed. Finishing up and testing correct ordering now.")

    # if u_func isa Array
    u_vals = tuple((reshape(u_vals[i], n_sensors[i], :) for i in 1:udim)...)
    # else
    #     u_vals = reshape(u_vals, n_sensors, :)
    # end
    v_vals = reshape(v_vals, vdim, :)
    y_locs = reshape(y_locs, ydim, :)
    seeds = reshape(seeds, :)

    # Split into training, validation, and test sets
    data = ( (y_locs, u_vals), v_vals )

    # SHUFFLE HERE???
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
            # println(size(y_locs_test))
            # println(size(first_seeds))
            # println(size(first_output))
            #
            # println(y_locs_test[:,i])
            # println(first_seeds[i])
            # println(first_output[:,i])
            # println(v_func(y_locs_test[:,i], first_seeds[i]))
            #
            @assert all(abs.(v_func(y_locs_test[:,i], first_seeds[i]) .- first_output[:,i]) .<= 1e-5)
            # @assert v_func(y_locs_test[:,i], first_seeds[i])[] == first_output[i]

            # if u_func isa Array
                u_vals_this_seed = u_func(x_locs, first_seeds[i])
                if !(u_vals_this_seed isa Tuple)
                    u_vals_this_seed = (u_vals_this_seed,)
                end
                for u_idx in 1:udim
                    @assert all(abs.(u_vals_this_seed[u_idx] - u_vals_test[u_idx][:,i]) .<= 1e-5)

                end

                    # @assert u_func[u_idx](x_locs[u_idx], first_seeds[i]) == Float32.(u_vals_test[u_idx][:, i])

            # else
            #     @assert all(abs.(u_func(x_locs, first_seeds[i]) - u_vals_test[:,i]) .<= 1e-5)
            # end
        end
    end



    println("\e[2K","\e[1G","Data generation finished")

    return (train=train_loader,
            validation=validation_loader,
            test=test_loader)

end





function train!(model, data_loaders, params, loss_fun, opt, n_epochs, loss_train=nothing, loss_validation=nothing, verbose=2, loss_fun_val = nothing, aux_params=nothing)
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss

    if params isa Vector
        opt_tree=Optimisers.setup(opt, params);
    end

    if loss_fun_val == nothing
        loss_fun_val = loss_fun
    end

    loss(x...) = aux_params!=nothing ? loss_fun(x..., aux_params) : loss_fun(x...)
    loss_val(x...) = aux_params!=nothing ? loss_fun_val(x..., aux_params) : loss_fun_val(x...)



    if loss_train == nothing
        loss_train = fill(NaN,n_epochs)
    end
    if loss_validation == nothing
        loss_validation = fill(NaN,n_epochs)
    end
    if verbose >= 1
        training_progress = ProgressBar(1:n_epochs)
        set_description(training_progress, "Training DeepONet:")
    else
        training_progress = 1:n_epochs
    end
    for e in training_progress
        loss_t=0
        loss_v=0
        for (d,s) in data_loaders.train
            if params isa Vector
                gs = ReverseDiff.gradient(tuple(params...)) do p...
                    training_loss = loss(d,s,[p...])
                    return training_loss
                end
                Optimisers.update!(opt_tree,params,[gs...])
                loss_t+=training_loss.value
                restructure_update(model)
            elseif params isa Flux.Params
                gs = Flux.gradient(params) do
                    training_loss = loss(d,s)
                    return training_loss
                end
                Flux.update!(opt, params, gs)
                loss_t+=training_loss
            else
                error("Wrong parameter type provided")
            end
        end
        # Make sure we get the mean loss
        loss_train[e]=loss_t/length(data_loaders.train)


        # Compute mean validation loss
        for (d,s) in data_loaders.validation
            loss_v+=loss_val(d,s)
        end
        loss_validation[e]=loss_v/length(data_loaders.validation)
        if verbose >= 2
            set_multiline_postfix(training_progress, @sprintf("Training loss: %.3e\nValidation loss: %.3e", loss_train[e], loss_validation[e]))
            flush(stdout)
            flush(stderr)
        end

    end

    return loss_train, loss_validation
end





end
