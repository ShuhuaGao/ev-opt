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
    input_dict = Dict(k=>data[string(k)] for k in fieldnames(EVData))
    return EVData(; input_dict...)
end