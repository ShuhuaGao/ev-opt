using NPZ


Base.@kwdef struct EVData
    ta::Vector{Int}         # included
    td::Vector{Int}         # excluded
    SoCa::Vector{Float64}
    SoCd::Vector{Float64}
    C::Vector{Float64}          # capacity
    e::Vector{Float64}          # efficiency of charging
    SoCmin::Vector{Float64}
    SoCmax::Vector{Float64}
    Pmax::Vector{Float64}
end


function EVData(npz_file::String)
    data = NPZ.npzread(npz_file)
    for key in ["rho", "delta_t", "b", "Plim"]
        delete!(data, key)
    end
    return EVData(; Dict(Symbol(k)=>v for (k, v) in data)...)
end