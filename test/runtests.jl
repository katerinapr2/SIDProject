using SIDProject
using Test

using JLD2
using TextAnalysis
using Transformers
using Transformers.HuggingFace

include("./../functions/toChunks.jl")
using .ToChunks
include("./../functions/preprocess.jl")


greek_tokenizer = hgf"nlpaueb/bert-base-greek-uncased-v1:tokenizer"

function checkLenChunks(start, final)
    return final .- start .<= 510
end

# # Δικαστική απόφαση 01/2018 
# # από το https://www.areiospagos.gr/nomologia/apofaseis.asp
# # με τυχαία μη πραγματικά δεδομένα.
doc = JLD2.load("text.jld2", "text").text;
targets = split(JLD2.load("targets.jld2", "trgs").text);

@testset "DocChunking" begin
    chunks, lenChunks, sI, eI = ToChunks.split_to_chunks(doc, greek_tokenizer)
    @test ones(Int, length(chunks)) == checkLenChunks(sI, eI)

    chunks, lenChunks, sI, eI = ToChunks.split_to_overlapping_chunks(doc, greek_tokenizer)
    @test ones(Int, length(chunks)) == checkLenChunks(sI, eI)
end 

@testset "TargetMatching" begin
    chunkTargets = []

    chunks, lenChunks, sI, eI = split_to_chunks(doc, greek_tokenizer)

    for (s, e) in zip(sI, eI)
        append!(chunkTargets, targets[s:e])
    end

    @test targets == chunkTargets
end

include("modeltests.jl")
