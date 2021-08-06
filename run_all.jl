using GaPLAC
using GaPLAC.ArgParse
using GaPLAC.CSV
using GaPLAC.DataFrames
using GaPLAC.LoggingExtras
using GaPLAC.TerminalLoggers
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

indir = normpath(expanduser(args["indir"]))
outpath = isnothing(args["outpath"]) ? nothing : normpath(expanduser(args["outpath"]))
pairs = args["pairs"]

if isnothing(pairs)
    pairsprinted = "all"
    @debug "Input pairs: all" 
elseif isfile(first(pairs))
    pairsprinted = "from file: $(normpath(expanduser(first(pairs))))"
    @debug "Input pairs: $(readlines(first(pairs)))" 
else
    firstpairs = length(pairs) > 5 ? first(pairs, 5) : pairs
    pairsprinted = string(join(firstpairs, ", "), "...")
    @debug "Input pairs: $pairs" 
end

@info """
    ## Running input diet pairs

    **Arguments:**

    - input directory: $indir
    - output file: $outpath
    - pairs: $pairsprinted
    - threads: $(Threads.nthreads())
    """

if !isnothing(pairs)
    if isfile(first(pairs))
        pairs = parse.(Int, readlines(first(pairs)))
    else
        pairs = parse.(Int, pairs)
    end
    pairs = Set(pairs)
end


files = readdir(args["indir"])
filter!(files) do file
    m = match(r"input_pair_(\d+)", file)
    isnothing(m) && return false
    isnothing(pairs) && return true
    return in(parse(Int, m.captures[1]), pairs)
end

@debug "Input files: $files"

outdf = DataFrame(file = files)
outdf.pair = map(file-> parse(Int, match(r"input_pair_(\d+)", file).captures[1]), outdf.file)
outdf.model1_logpdf = zeros(size(outdf, 1))
outdf.model2_logpdf = zeros(size(outdf, 1))
outdf.log2bayes = zeros(size(outdf, 1))

@threads for (i, file) in outdf.file
    df = CSV.read(file, DataFrame)
    df = disallowmissing(df[completecases(df), :])

    k_t = SqExponentialKernel()
    k_sub = CategoricalKernel()
    k_diet = LinearKernel()

    k1 = (k_t ⊗ k_sub) ∘ SelectTransform([1,2]) + k_diet ∘ SelectTransform([3]) # Collect all the kernels to make them act dimension wise
    k2 = (k_t ⊗ k_sub) ∘ SelectTransform([1,2]) # kernel without diet variable

    ##

    # Here we create a the prior based on the kernel and the data
    pr1 = AbstractGPs.FiniteGP(GP(k1), hcat(df.timepoint, df.subject, df.diet), 0.1, obsdim = 1)
    pr2 = AbstractGPs.FiniteGP(GP(k2), hcat(df.timepoint, df.subject), 0.1, obsdim = 1)

    # We finally compute the posterior given the y observations

    pst1 = pstior(pr1, df.abundance_norm)
    pst2 = pstior(pr2, df.abundance_norm)

    p1 = logpdf(pr1, df.abundance_norm)
    p2 = logpdf(pr2, df.abundance_norm)

    l2bayes = log2(BigFloat(2)^logpdf(pr1, df.abundance_norm) / 
        BigFloat(2)^logpdf(pr2, df.abundance_norm) 
    )

    outdf[i, :model1_logpdf] = p1
    outdf[i, :model2_logpdf] = p2
    outdf[i, :log2bayes] = l2bayes
end

GaPLAC._df_output(outdf, args)