using Flux
using Flux.Data: DataLoader
using CUDA
using NeuralAttentionlib: LengthMask

# include("./../src/models.jl")

SIDLabels = ["No", "Yes"]
BatchSize = 2

doc = StringDocument(doc)
targets = StringDocument(join(targets, " "))

X_test, y_test = inputs([doc], [targets], SIDLabels, greek_tokenizer, 0)

test_data = (X_test.token, X_test.segment, X_test.attention_mask.len, y_test)
test_data_loader = DataLoader(test_data; batchsize=BatchSize, shuffle=false)

model = create_model(length(SIDLabels), 0.1) |> gpu

Xy = collect(enumerate(test_data_loader))[1][2]
input = (token=Xy[1], segment=Xy[2], attention_mask=LengthMask{1, Vector{Int32}}(Xy[3])) |> gpu

output = model(input, [[i for i in 2:(Xy[3][j]-1)] for j in 1:BatchSize]) |> cpu

greek_bert = hgf"nlpaueb/bert-base-greek-uncased-v1:model" |> gpu
bert_output = greek_bert(input).hidden_state |> cpu
filtered = rejectSpecialTokens(bert_output, Xy[3])

# Check whether the classifier is applied only on tokens (not the special ones)
@testset "ModelOutput" begin
    @test size(output, 2) == sum(Xy[3].-2)
    @test filtered[:,end-10:end] == bert_output[:, (Xy[3][end]-11):(Xy[3][end]-1), end]
end
