############ Model without NER ############
struct Model_withoutNER{B, C}
    bert::B
    classifier::C
end

function rejectSpecialTokens(logits, attention_mask)
    out = logits[:, 2:attention_mask[1]-1, 1] 
    for b in 2:size(logits, 3)
        out = hcat(out, logits[:, 2:attention_mask[b]-1, b])
    end
    return out
end


function (self::Model_withoutNER)(input, attention_mask)
    logits = self.bert(input).hidden_state
    filtered_logits = rejectSpecialTokens(logits, attention_mask)
    return (
        self.classifier(filtered_logits)
    )
end

Flux.@functor Model_withoutNER

function create_model_withoutNER(nlabel, d)
    greek_bert = hgf"nlpaueb/bert-base-greek-uncased-v1:model"
    cfg = hgf"nlpaueb/bert-base-greek-uncased-v1:config"
    dims= cfg[:hidden_size]
    cls = Chain(
            Dropout(d), 
            Dense(dims, 256, relu),
            # Dropout(d), 
            Dense(256, 1, sigmoid)  
            )

    return Model_withoutNER(greek_bert, cls)
end



############ Model with NER ############
struct Model{B, C}
    bert::B
    classifier::C
end

function keepChosenTokens(logits, nerIndexes)
    out = []
    for i in 1:length(nerIndexes)
        token = nerIndexes[i] .+ 1
        if !isempty(token) & isempty(out)
            out = logits[:, token,i]
        elseif !isempty(token) & !isempty(out)
            out = hcat(out, logits[:, token, i])
        end
    end
    return out
end


function (self::Model)(input, nerIndexes)
    logits = self.bert(input).hidden_state
    filtered_logits = keepChosenTokens(logits, nerIndexes)
    return (
        self.classifier(filtered_logits)
    )
end

Flux.@functor Model

function create_model(nlabel, d)
    greek_bert = hgf"nlpaueb/bert-base-greek-uncased-v1:model"
    cfg = hgf"nlpaueb/bert-base-greek-uncased-v1:config"
    dims= cfg[:hidden_size]
    cls = Chain(
            Dropout(d), 
            Dense(dims, 256, relu),
            # Dropout(d), 
            Dense(256, 1, sigmoid)
            )

    return Model(greek_bert, cls)
end















# struct SIDModel{B, C}
#     bert::B
#     classifier::C
# end

# Flux.@functor SIDModel

# (self::SIDModel)(input;
#                 position_ids = nothing, token_type_ids = nothing,
#                 attention_mask = nothing,
#                 output_attentions = false,
#                 output_hidden_states = false
#                 ) = self(input, position_ids, token_type_ids,
#                         attention_mask,
#                         Val(output_attentions), Val(output_hidden_states))

# function (self::SIDModel)(input, position_ids, token_type_ids,
#                                                 attention_mask,
#                                                 _output_attentions::Val{output_attentions},
#                                                 _output_hidden_states::Val{output_hidden_states}
#                                                 ) where {output_attentions, output_hidden_states}
#     outputs = self.bert(input, position_ids, token_type_ids,
#                 attention_mask, _output_attentions, _output_hidden_states)
#     sequence_output = outputs[1]
#     logits = self.classifier(sequence_output)
#     loss = nothing

#     return (
#     loss = loss,
#     logits = logits,
#     hidden_states = outputs.hidden_states,
#     attentions = outputs.attentions
#     )
# end

# (self::SIDModel)(x::Tuple) = self(x[1]; token_type_ids=x[2], attention_mask=x[3])



# function create_SIDModel(nlabel)
#     greek_bert = hgf"nlpaueb/bert-base-greek-uncased-v1:model"
#     cfg = hgf"nlpaueb/bert-base-greek-uncased-v1:config"
#     dims= cfg[:hidden_size]
#     cls = Chain(
#             Dropout(0.1), 
#             Dense(dims, nlabel),
#             softmax
#             )
#     return SIDModel(greek_bert, cls)
# end

