using Documenter, Devito

makedocs(sitename="Devito", modules=[Devito])

deploydocs(
    repo = "github.com/ChevronETC/Devito.jl.git",
)