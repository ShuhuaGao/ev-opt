# fairness is squared as the objective item

using JuMP, Ipopt
using AmplNLWriter, Couenne_jll, SHOT_jll, Bonmin_jll
using KNITRO
using Dates


include("EV.jl")

"""
    compute_objective

Given a decision vector `a`, compute the objective value of the optimization problem.
"""
function compute_objective(a::AbstractMatrix{Float64}, EVs::EVData, ρ::AbstractVector{Float64}; 
    Δt::Float64=1.0, β::Float64)
    K, T = size(a)
    c_obj = f_obj = 0.0
    P = a .* EVs.Pmax
    # first compute SoC
    SoC = zeros(Float64, K, T+1)
    SoC[:, 1] .= EVs.SoCa
    for t = 1:T, k = 1:K
        # if EV does not stay here, then a must be zero and will affect the result
        ΔSoC = P[k, t] * EVs.e[k] * Δt / EVs.C[k] 
        SoC[k, t+1] = SoC[k, t] + ΔSoC
        c_obj += P[k, t] * Δt * ρ[t]
    end

    ta_min = minimum(EVs.ta)
    for t = 1:T
        τ = t + ta_min - 1   # actual time without shifting
        gt = zeros(K)
        for k = 1:K
            if EVs.ta[k] <= τ < EVs.td[k] # ev arrived and not leaved
                gt[k] = (EVs.SoCd[k] - SoC[k, t]) * EVs.C[k] / (EVs.Pmax[k] * EVs.e[k] * Δt) / (EVs.td[k] - τ)
            end
        end
        gt_total = sum(gt)
        Gt = zeros(K)
        if gt_total > 0
            for k = 1:K
                Gt[k] = gt[k] / gt_total
            end
        end
        Pt_total = sum(P[:, t])
        for k = 1:K
            f_obj += (P[k, t] - Pt_total * Gt[k])^2
        end
    end
    return (1 - β) * c_obj + β * f_obj
end



"""
    build_model(EVs::EVData, ρ::AbstractVector{Float64}; Δt::Float64, β::Float64, Plim::Float64)

Build an optimization model for EV charging.
The price vector `ρ` corresponds to time range [ta_min, td_max - 1] among all EVs.
"""
function build_model(EVs::EVData, ρ::AbstractVector{Float64}; Δt::Float64=1.0, β::Float64, Plim::Float64)
    all(td > ta for (ta, td) in zip(EVs.ta, EVs.td)) || error("Invalid time range")
    ta_min = minimum(EVs.ta)
    td_max = maximum(EVs.td)
    T = td_max - ta_min     # because td is excluded 
    # @show T length(ρ)
    @assert length(ρ) == T
    @assert 0 <= β <= 1
    @assert Plim > 0
    # shift the time to make ta_min = 1
    EVs = deepcopy(EVs)   # avoding changing the argument passed in 
    EVs.ta .-= (ta_min - 1)
    EVs.td .-= (ta_min - 1)
    K = length(EVs.SoCmin)    # K vehicles

    model = JuMP.Model()

    @variable(model, 0 <= a[1:K, 1:T] <= 1.0)
    @expression(model, P, a .* EVs.Pmax)        # K×T
    # power before ta or after and on td must be enforced to be 0
    for k = 1:K
        fix.(a[k, 1:EVs.ta[k]-1], 0; force=true)
        fix.(a[k, EVs.td[k]:T], 0; force=true)
    end

    # SoC[k, t] is the state of charge of EV k at the **beginning** of time step t
    @expression(model, SoC[1:K, 1:T+1], zero(AffExpr))
    # SoC dynamics
    SoC[:, 1] .= EVs.SoCa
    for t = 1:T
        SoC[:, t+1] .= @expression(model, SoC[:, t] .+ P[:, t] .* EVs.e .* Δt ./ EVs.C)
    end
    # SoC[:, 1] .= @expression(model, EVs.SoCa .+ P[:, 1] .* EVs.e .* Δt ./ EVs.C)
    # for t = 2:T
    #     SoC[:, t] .= @expression(model, SoC[:, t-1] .+ P[:, t] .* EVs.e .* Δt ./ EVs.C)
    # end
    # SoC constraints
    @constraint(model, EVs.SoCmin .<= SoC .<= EVs.SoCmax)
    for k = 1:K
        @constraint(model, SoC[k, EVs.td[k]] == EVs.SoCd[k])
    end

    # station power limit
    @constraint(model, sum(P; dims=1) .<= Plim)

    # cost objective
    @expression(model, c_obj, ρ' .* P .* Δt)     # K × T
    # fairness objective
    @expression(model, g[1:K, 1:T], zero(AffExpr))
    for k = 1:K
        ts = EVs.ta[k]:EVs.td[k]-1  # valid time range for EV k 
        g[k, ts] .= EVs.C[k] .* (EVs.SoCd[k] .- SoC[k, ts]) ./ (EVs.Pmax[k] .* Δt .* EVs.e[k])  ./ (EVs.td[k] .- ts)
    end
    @expression(model, G, g ./ (sum(g; dims=1) .+ 1e-8))  # K × T, 1e-8 avoids zero division 
    @expression(model, f_obj, (P .- sum(P; dims=1) .* G).^2)  # K × T

    # total objective: weighted sum 
    @objective(model, Min, sum((1 - β) .* c_obj .+ β .* f_obj))
    
    return model
end


function solve!(model::JuMP.Model; optimizer::Symbol=:Couenne)
    if optimizer == :SCIP
        set_optimizer(model, SCIP.Optimizer)
    elseif optimizer == :Alpine
        ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
        highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
        set_optimizer(model, optimizer_with_attributes(
            Alpine.Optimizer,
            "nlp_solver" => ipopt,
            "mip_solver" => highs,
        ))
    elseif optimizer == :Ipopt
        set_optimizer(model, Ipopt.Optimizer)
        set_attribute(model, "tol", 1e-6)
        set_attribute(model, "acceptable_tol", 1e-4)
    elseif optimizer == :Couenne
        set_optimizer(model, () -> AmplNLWriter.Optimizer(Couenne_jll.amplexe))
    elseif optimizer == :SHOT
        set_optimizer(model, () -> AmplNLWriter.Optimizer(SHOT_jll.amplexe))
    elseif optimizer == :Bonmin
        set_optimizer(model, () -> AmplNLWriter.Optimizer(Bonmin_jll.amplexe))
    elseif optimizer == :KNITRO
        @show KNITRO.has_knitro()
        set_optimizer(model, () -> AmplNLWriter.Optimizer(KNITRO.amplexe, 
            ["alg=0",                   # select optimization algorithm, 0: auto 
             "maxit=200000",            # max iterations
             "maxtime_real=6000",       # max time
             "opttol=5e-2",             # optimality relative tolerance
             "opttol_abs=10",           # optimality absolute tolerance
             "feastol=1e-3",            # feasibility relative tolerance
             "feastol_abs=1e-2",         # feasibility absolute tolerance 
             "outlev=1",
             "ms_enable=1",             # multistart: https://www.artelys.com/app/docs/knitro/2_userGuide/multistart.html
             "ms_maxsolves=10",
             "ms_outsub=1"     
             ]))
    else
        error("Unsupported optimizer")
    end

    optimize!(model)

    @show termination_status(model)
    if termination_status(model) ∈ [JuMP.INFEASIBLE, JuMP.INFEASIBLE_OR_UNBOUNDED]
        @error "Infeasible"
    end
    @show objective_value(model)

    return model
end


function run(npz_file::String; optimizer::Symbol=:Couenne)
    EVs = EVData(npz_file)
    cfg = npzread(npz_file)
    Δt=cfg["delta_t"]
    β=cfg["beta"]
    ρ = cfg["rho"]
    model = build_model(EVs, ρ; Δt, β, Plim=cfg["Plim"])
    try
        solve!(model; optimizer)
        # store results
        keys = ["a", "P", "g", "G", "SoC", "c_obj", "f_obj"]
        res = Dict{String, Union{Matrix{Float64}, Float64}}(k=>value.(model[Symbol(k)]) for k in keys)
        res["obj_value"] = objective_value(model)
        target_npz_file = npz_file[1:end-4] * "-res-$(optimizer)-2.npz"
        println("Optimization towards $npz_file finished! Checking feasibility...")
        fr_dict = primal_feasibility_report(model, atol=1e-3)
        if isempty(fr_dict)
            printstyled("Feasible solution found and written to $target_npz_file.\n"; color=:green)
            npzwrite(target_npz_file, res)
            println("Evaluate objective...")
            eval_obj = compute_objective(res["a"], EVs, ρ; Δt, β)
            println("opt_obj / eval_obj = $(res["obj_value"]) / $eval_obj. Difference = $(res["obj_value"] - eval_obj)")
        else
            printstyled("Infeasible. $(length(fr_dict)) constraints violated.\n"; color=:red)
        end
    catch e
        println("Unable to optimize $npz_file ! Error: $e")
    end
end


function main(npz_files::AbstractVector{String}; optimizer::Symbol=:Couenne)
    @show Threads.nthreads()        # number of threads available
    Threads.@threads for npz_file in npz_files
        run(npz_file; optimizer)
    end
end


# process a list of input data in parallel
main(["./data/EVDATA-t0.25-b0.1.npz"]; optimizer=:KNITRO)
