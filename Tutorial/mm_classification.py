def multimodal_classification(ecg_shape, text_vocab_size, max_text_length, num_classes):
    ecg_input = Input(shape=ecg_shape)
    x = Conv1D(64, kernel_size=5, activation='relu')(ecg_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    ecg_features = GlobalAveragePooling1D()(x)

    # input text
    text_input = Input(shape=(max_text_length,))
    embedded_text = Embedding(input_dim=text_vocab_size, output_dim=64, input_length=max_text_length)(text_input)
    
    # BiLSTM 
    bilstm_out = Bidirectional(LSTM(64))(embedded_text)

    # ECG features with BiLSTM output
    combined = Concatenate()([ecg_features, bilstm_out])
    
    x = Dense(128, activation='relu')(combined) 
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[ecg_input, text_input], outputs=output)
    
    return model
