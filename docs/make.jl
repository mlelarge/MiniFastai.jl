using MiniFastai
using Documenter

makedocs(;
    modules=[MiniFastai],
    authors="Marc Lelarge",
    repo="https://github.com/mlelarge/MiniFastai.jl/blob/{commit}{path}#L{line}",
    sitename="MiniFastai.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mlelarge.github.io/MiniFastai.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mlelarge/MiniFastai.jl",
)
