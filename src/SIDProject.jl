module SIDProject

using BSON
using JLD2
using TextAnalysis
using AbstractTrees
using StringEncodings
using Transformers
using Transformers.HuggingFace
using Flux
using CUDA
using NeuralAttentionlib
using NeuralAttentionlib: LengthMask
using Unicode: normalize
using CondaPkg
using PythonCall

include(".\\..\\functions\\toChunks.jl")
using .ToChunks

include(".\\..\\functions\\preprocess.jl")
include(".\\..\\functions\\ner\\ner.jl")

include("models.jl")
include("anonymization.jl")

export create_model, anonymize, rejectSpecialTokens

end 