@info "Getting Started"

using GaPLAC
using LoggingExtras
using TerminalLoggers
using ArgParse
using GaPLAC.CSV
using GaPLAC.DataFrames
using GaPLAC.AbstractGPs
using GaPLAC.KernelFunctions

function parse_cmdline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "indir"
            help = "Path to directory containing input pair tables"
        "--pairs", "-p"
            help = "Either a file containing pair #s, or a space-separated list of numbers"
            nargs = '+'
        "--output", "-o"
            help = "Path to file for table output"
        "--verbose", "-v"
            help = "Log level to @info"
            action = :store_true
        "--quiet", "-q"
            help = "Log level to @warning"
            action = :store_true
        "--debug"
            help = "Log level to @debug"
            action = :store_true
        "--log"
            help = "Log to a file as well as stdout"

    end

    return parse_args(s)
end

args = parse_cmdline()

function setup_logs!(loglevel, logpath; dryrun=false)
    glog = TerminalLogger(stderr, loglevel)
    if logpath === nothing || dryrun
        global_logger(glog)
    else
        logpath = abspath(expanduser(logpath))
        global_logger(
            TeeLogger(
                MinLevelLogger(FileLogger(logpath), loglevel),
                glog))
    end
end

if args["debug"]
    loglevel = Logging.Debug
elseif args["verbose"]
    loglevel = Logging.Info
elseif args["quiet"]
    loglevel = Logging.Error
else
    loglevel = Logging.Warn
end

setup_logs!(loglevel, args["log"])

@debug args

indir = normpath(expanduser(args["indir"]))
outpath = isnothing(args["output"]) ? nothing : normpath(expanduser(args["output"]))
inpairs = args["pairs"]

if isempty(inpairs)
    pairsprinted = "all"
    @debug "Input pairs: all" 
elseif isfile(first(inpairs))
    pairsprinted = "from file: $(normpath(expanduser(first(pairs))))"
    @debug "Input pairs: $(readlines(first(pairs)))" 
else
    firstpairs = length(pairs) > 5 ? first(inpairs, 5) : inpairs
    pairsprinted = string(join(firstpairs, ", "), "...")
    @debug "Input pairs: $inpairs" 
end

@info """
    ## Running input diet pairs

    **Arguments:**

    - input directory: $indir
    - output file: $outpath
    - pairs: $pairsprinted
    - threads: $(Threads.nthreads())
    """

if !isempty(inpairs)
    if isfile(first(inpairs))
        inpairs = parse.(Int, readlines(first(inpairs)))
    else
        inpairs = parse.(Int, inpairs)
    end
    inpairs = Set(inpairs)
end


files = readdir(args["indir"])
@debug files
filter!(files) do file
    @debug "filtering"
    @debug file
    m = match(r"input_pair_(\d+)", file)
    @debug m
    isnothing(m) && return false
    isempty(inpairs) && return true
    return in(parse(Int, m.captures[1]), inpairs)
end

@debug "Input files: $files"

outdf = DataFrame(file = files)
outdf.pair = map(file-> parse(Int, match(r"input_pair_(\d+)", file).captures[1]), outdf.file)
outdf.model1_logpdf = zeros(size(outdf, 1))
outdf.model2_logpdf = zeros(size(outdf, 1))
outdf.log2bayes = zeros(size(outdf, 1))

Threads.@threads for (i, file) in enumerate(outdf.file)
    df = CSV.read(joinpath(indir, file), DataFrame)
    !in("Date", names(df)) && continue

    df = disallowmissing(df[completecases(df), :])

    k_t = SqExponentialKernel()
    k_sub = GaPLAC.CategoricalKernel()
    k_diet = LinearKernel()

    k1 = (k_t ⊗ k_sub) ∘ SelectTransform([1,2]) + k_diet ∘ SelectTransform([3]) # Collect all the kernels to make them act dimension wise
    k2 = (k_t ⊗ k_sub) ∘ SelectTransform([1,2]) # kernel without diet variable

    ##

    # Here we create a the prior based on the kernel and the data
    pr1 = AbstractGPs.FiniteGP(GP(k1), hcat(df.Date, df.PersonID, df.nutrient), 0.1, obsdim = 1)
    pr2 = AbstractGPs.FiniteGP(GP(k2), hcat(df.Date, df.PersonID), 0.1, obsdim = 1)

    # We finally compute the posterior given the y observations

    pst1 = posterior(pr1, df.bug)
    pst2 = posterior(pr2, df.bug)

    p1 = logpdf(pr1, df.bug)
    p2 = logpdf(pr2, df.bug)

    l2bayes = log2(BigFloat(2)^logpdf(pr1, df.bug) / 
                   BigFloat(2)^logpdf(pr2, df.bug) 
    )

    outdf[i, :model1_logpdf] = p1
    outdf[i, :model2_logpdf] = p2
    outdf[i, :log2bayes] = l2bayes
end

GaPLAC._df_output(outdf, args)