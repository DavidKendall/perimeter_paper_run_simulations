module ModelStats

using Statistics

include("model.jl")
import .SwarmModel as SM
using .SwarmModel

export knn_mean_distances, knn_d

function agent_perimeter_status(b, parameters; n_steps=500, boundary=50)
    n_agents = size(b)[1]
    accum = zeros(n_agents, n_steps)
    b_, parameters_ = deepcopy(b), deepcopy(parameters)
    for i in 1:n_steps
        compute_step(b_; parameters_...)
        accum[:, i] .= b_[:, SM.PRM]
        apply_step(b_)
    end
    return vec(Int.((sum(accum, dims=2) ./ n_steps .* 100.0) .> boundary) .+ 1)
end

function knn_d(k, mag, cb, p, perim_pair, nbr_mag)
    n_agents, max_n = size(nbr_mag)
    nbr_mag .= 0.
    Threads.@threads for i in 1:n_agents
        n = 0
        for j in 1:n_agents
            if (p[i], p[j]) == perim_pair && mag[j,i] ≤ cb
                n += 1
                nbr_mag[i, n] = mag[j, i]
            end
        end
        sort!(@view(nbr_mag[i, 1:n]))
        if k < n
            nbr_mag[i, k+1 : n] .= 0.
        end
    end
    not_zero = nbr_mag .!= 0.
    return mean(nbr_mag[not_zero]), std(nbr_mag[not_zero])
end

function knn_mean_distances(b, parameters; n_steps=500, class_ids=[:ii, :pi, :pp], k=[2,1,1,2], perimeter=nothing, failure=nothing)
    id_to_int = Dict(:ii => 1, :ip => 2, :pi => 3, :pp => 4)
    id_to_perim_pair = Dict(:ii => (1, 1), :ip => (1, 2), :pi => (2, 1), :pp => (2, 2))
    n_classes = length(class_ids)
    means = Array{Float64}(undef, n_steps, n_classes)
    stds = Array{Float64}(undef, n_steps, n_classes)
    if failure !== nothing
        f_step, failed = failure
    end
    for i in 1:n_steps
        n_agents = size(b)[1]
        xv, yv, mag, p = compute_step(b; parameters...)
        if perimeter !== nothing
            p = perimeter
        end
        max_n = Int(maximum(b[:, SM.COH_N]))
        nbr_mag = Array{Float64}(undef, n_agents, max_n)
        cn = 0
        for c in class_ids
            cn += 1
            cl = id_to_int[c]
            pp = id_to_perim_pair[c]
            m, s = knn_d(k[cl], mag, parameters[:cb], p, pp, nbr_mag)
            means[i, cn] = m
            stds[i, cn] = s
        end
        apply_step(b)
        if failure != nothing && i == f_step
            b_ = [b[a,:] for a in 1:n_agents if a ∉ failed]
            b = collect(transpose(reshape(collect(Iterators.flatten(b_)),(SM.N_COLS, n_agents - length(failed)))))
        end
    end
    return means, stds
end

function centroid(b)
    return mean(b[:,SM.POS_X]), mean(b[:,SM.POS_Y])
end

function radius(b)
    c = centroid(b)
    prm = findall(b[:,SM.PRM] .> 0.)
    mean(hypot.(b[prm, SM.POS_X] .- c[1], b[prm, SM.POS_Y] .- c[2]))
end

function area(b)
    return π * radius(b) ^ 2
end

function density(b)
    return size(b)[1] / area(b)
end
    
end # module
