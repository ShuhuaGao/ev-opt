# ev-opt
Optimization for EV charging.

## How to use
1. Set the [`JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS) environment variable on your system to a proper number (usually the number of cores). 
   Note that by default its value is 1, that is, only one core is used during Julia execution even if you do parallel computing.
2. Start Julia REPL, and set up the environment for ev-opt in Julia REPL. Recall the usage of `cd()`, `]`, `activate .`  and `update`
3. Edit the "nlp.jl" file to change the inputs to the `main` call. Besides, you can also specify the optimizer, either `:Couenne` or `:Ipopt`.
4. Edit the Couenne option file "couenne.opt"
5. In Julia REPL, run `include("nlp.jl")`.
6. Once finished, the result of *input.npz* data is stored in *input-res.npz*.
