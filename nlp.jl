using JuMP, SCIP, Ipopt, EAGO
using Alpine, HiGHS

include("EV.jl")

"""
    build_model(EVs::EVData, ρ::AbstractVector{Float64}; Δt::Float64, β::Float64, Plim::Float64)

Build an optimization model for EV charging.
The price vector `ρ` corresponds to time range [ta_min, td_max - 1] among all EVs.
"""
function build_model(EVs::EVData, ρ::AbstractVector{Float64}; Δt::Float64, β::Float64, Plim::Float64)
    all(td > ta for (ta, td) in zip(EVs.ta, EVs.td)) || error("Invalid time range")
    ta_min = minimum(EVs.ta)
    td_max = maximum(EVs.td)
    T = td_max - ta_min     # because td is excluded 
    @show T length(ρ)
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

    # SoC[k, t] is the state of charge of EV k at the end of time step t
    @expression(model, SoC[1:K, 1:T], zero(AffExpr))
    # SoC dynamics
    SoC[:, 1] .= @expression(model, EVs.SoCa .+ P[:, 1] .* EVs.e .* Δt ./ EVs.C)
    for t = 2:T
        SoC[:, t] .= @expression(model, SoC[:, t-1] .+ P[:, t] .* EVs.e .* Δt ./ EVs.C)
    end
    # SoC constraints
    @constraint(model, EVs.SoCmin .<= SoC .<= EVs.SoCmax)
    for k = 1:K
        @constraint(model, SoC[k, EVs.td[k] - 1] == EVs.SoCd[k])
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
    @expression(model, f_obj, abs.(P .- sum(P; dims=1) .* G))  # K × T

    # total objective: weighted sum 
    @objective(model, Min, sum((1 - β) .* c_obj .+ β .* f_obj))

    # uncomment the following lines to move the nonlinear `f_obj` into constraints
    # @variable(model, f_obj_aux)
    # @constraint(model, sum(f_obj) <= f_obj_aux)
    # @objective(model, Min, (1 - β) * sum(c_obj) + β * f_obj_aux)
    
    return model
end


function solve!(model::JuMP.Model, optimizer::Symbol=:SCIP)
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
    else
        error("Unsupported optimizer")
    end
    optimize!(model)

    if termination_status(model) == JuMP.OPTIMAL
        return model
    else
        error("Optimization failed: $(termination_status(model))")
    end
    
end


function main()
    EVs = EVData("./data/EVDATA.npz")
    cfg = npzread("./data/EVDATA.npz")
    model = build_model(EVs, cfg["rho"]; Δt=cfg["delta_t"], β=cfg["b"], Plim=cfg["Plim"])
    solve!(model, :SCIP)
end


main()